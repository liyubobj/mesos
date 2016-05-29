// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <set>
#include <vector>

#include <process/dispatch.hpp>
#include <process/future.hpp>
#include <process/once.hpp>

#include <stout/option.hpp>
#include <stout/try.hpp>

#include "slave/flags.hpp"

#include "slave/containerizer/mesos/isolators/gpu/allocator.hpp"
#include "slave/containerizer/mesos/isolators/gpu/nvml.hpp"

using process::Failure;
using process::Future;
using process::Once;

using std::set;
using std::vector;

namespace mesos {
namespace internal {
namespace slave {

static constexpr unsigned int NVIDIA_MAJOR_DEVICE = 195;


// Build the resource vector of GPUs we manage. We determine this from
// inspecting the values of the `gpus` resource flag and the
// `nvidia_gpu_devices` flag. There are two cases to consider: either
// both flags are set or neither flag is set. If both flags are set,
// we need to make sure the value in the `gpus` resource is equal to
// the number of items in the `nvidia_gpu_devices` flag. We then
// just return the `gpus` resource specified. If neither flag is set,
// we look up the number of GPUs on the machine using the Nvidia
// Management Library (NVML) and build our resource vector from that.
// We special case `gpus:0` to not allow setting `nvidia_gpu_devices`
// for obvious reasons.
static Try<Resources> enumerateResources(const Flags& flags)
{
  Try<Resources> parsed = Resources::parse(
      flags.resources.getOrElse(""), flags.default_role);

  if (parsed.isError()) {
    return Error(parsed.error());
  }

  Resources resources = parsed.get().filter(
      [](const Resource& resource) {
          return resource.name() == "gpus";
      });

  // Using the resource vector alone, there is no way to
  // distinguish between the case of setting `gpus:0` in the
  // `resources` flag and not setting `gpus` at all. To help with
  // this we short circuit for the case of `gpus:0` and return an
  // empty resource vector. By doing so, we know from here on out
  // that checking `resources.gpus().isSome()` is sufficient to
  // determine if a value was set for `gpus`.
  if (strings::contains(flags.resources.getOrElse(""), "gpus") &&
      !resources.gpus().isSome()) {
    if (flags.nvidia_gpu_devices.isSome()) {
      return Error("The `--nvidia_gpus_devices` flag cannot be"
                   " specified when the `gpus` resource is set to 0");
    }
    return Resources();
  }

  if (flags.nvidia_gpu_devices.isSome() && !resources.gpus().isSome()) {
    return Error("The `--nvidia_gpus_devices` flag cannot be set"
                 " without also setting the `gpus` resource");
  }

  if (resources.gpus().isSome() && !flags.nvidia_gpu_devices.isSome()) {
    return Error("The `gpus` resource can not be set without also"
                 " setting the `--nvidia_gpu_devices` flag");
  }

  // If the `nvidia_gpu_devices` flag is set, make sure it contains a
  // list of unique GPU identifiers.
  if (flags.nvidia_gpu_devices.isSome()) {
    vector<unsigned int> unique = flags.nvidia_gpu_devices.get();
    std::sort(unique.begin(), unique.end());
    auto last = std::unique(unique.begin(), unique.end());
    unique.erase(last, unique.end());

    if (unique.size() != flags.nvidia_gpu_devices->size()) {
      return Error("The `nvidia_gpu_devices` flag must contain"
                   " a list of unique GPU identifiers");
    }
  }

  // Initialize NVML and grab the total GPU
  // count from it. Both code paths below require it.
  Try<Nothing> nvml = nvml::initialize();
  if (nvml.isError()) {
    return Error(nvml.error());
  }

  Try<unsigned int> total = nvml::deviceGetCount();
  if (total.isError()) {
    return Error(total.error());
  }

  // If the `gpus` resource is set, verify that its value is valid,
  // make sure there are enough GPUs on the machine to satisfy it, and
  // return the given resource vector.
  if (resources.gpus().isSome()) {
    // Make sure that the value of `gpus` is an integer and not a
    // fractional amount. We take advantage of the fact that we know
    // the value of `gpus` is only precise up to 3 decimals.
    long long milli = static_cast<long long>(resources.gpus().get() * 1000);
    if ((milli % 1000) != 0) {
      return Error("The `gpus` resource must be an unsigned integer");
    }

    if (flags.nvidia_gpu_devices->size() != resources.gpus().get()) {
      return Error("The number of GPUs passed in the '--nvidia_gpu_devices'"
                   " flag must match the number of GPUs specified in the"
                   "'gpus' resource");
    }

    if (flags.nvidia_gpu_devices->size() > total.get()) {
      return Error("The number of GPUs requested is greater than"
                   " the number of GPUs available on the machine");
    }

    return resources;
  }

  // If the `gpus` resource is not set, build a resource
  // vector from the GPU device count and return it.
  return Resources::parse(
      "gpus",
      stringify(total.get()),
      flags.default_role).get();
}


// Build the `set` of GPUs to manage. We determine this set from
// inspecting the incoming resource vector as well as the
// `nvidia_gpu_devices` flag.
static Try<set<Gpu>> enumerateGpus(
    const Flags& flags,
    const Resources& resources)
{
  if (!resources.gpus().isSome()) {
    return set<Gpu>();
  }

  vector<unsigned int> indices;
  if (flags.nvidia_gpu_devices.isSome()) {
    indices = flags.nvidia_gpu_devices.get();
  } else {
    for (unsigned int i = 0; i < resources.gpus().get(); ++i) {
      indices.push_back(i);
    }
  }

  set<Gpu> gpus;
  foreach (unsigned int index, indices) {
    Try<nvmlDevice_t> handle = nvml::deviceGetHandleByIndex(index);
    if (handle.isError()) {
      return Error(handle.error());
    }

    Try<unsigned int> minor = nvml::deviceGetMinorNumber(handle.get());
    if (minor.isError()) {
      return Error(minor.error());
    }

    Gpu gpu;
    gpu.major = NVIDIA_MAJOR_DEVICE;
    gpu.minor = minor.get();

    gpus.insert(gpu);
  }

  return gpus;
}


Try<NvidiaGpuAllocator*> NvidiaGpuAllocator::create(const Flags& flags)
{
  Try<Resources> resources = enumerateResources(flags);
  if (resources.isError()) {
    return Error(resources.error());
  }

  Try<set<Gpu>> gpus = enumerateGpus(flags, resources.get());
  if (gpus.isError()) {
    return Error(gpus.error());
  }

  return new NvidiaGpuAllocator(gpus.get());
}


Try<Resources> NvidiaGpuAllocator::resources(const Flags& flags)
{
  return enumerateResources(flags);
}


NvidiaGpuAllocator::NvidiaGpuAllocator(
    const set<Gpu>& _gpus)
  : gpus(_gpus),
    process(new NvidiaGpuAllocatorProcess(_gpus))
{
  spawn(process.get());
}


NvidiaGpuAllocator::~NvidiaGpuAllocator()
{
  terminate(process.get());
  process::wait(process.get());
}


const set<Gpu>& NvidiaGpuAllocator::allGpus() const
{
  return gpus;
}


Future<Option<set<Gpu>>> NvidiaGpuAllocator::allocate(size_t count) const
{
  std::function<Future<Option<set<Gpu>>>()> f = [this, count] () {
      return process->allocate(count);
  };
  return dispatch(process->self(), f);
}

Future<Nothing> NvidiaGpuAllocator::allocate(const set<Gpu>& gpus) const
{
  std::function<Future<Nothing>()> f = [this, gpus] () {
      return process->allocate(gpus);
  };
  return dispatch(process->self(), f);
}


Future<Nothing> NvidiaGpuAllocator::deallocate(const set<Gpu>& gpus) const
{
  std::function<Future<Nothing>()> f = [this, gpus] () {
      return process->deallocate(gpus);
  };
  return dispatch(process->self(), f);
}


NvidiaGpuAllocatorProcess::NvidiaGpuAllocatorProcess(const set<Gpu>& gpus)
  : available(gpus) {}


Future<Option<set<Gpu>>> NvidiaGpuAllocatorProcess::allocate(size_t count)
{
  if (count == 0 || available.size() < count) {
    return None();
  }

  set<Gpu> gpus;
  set<Gpu> availableCopy = available;

  foreach (const Gpu& gpu, availableCopy) {
    available.erase(gpu);
    gpus.insert(gpu);
    taken.insert(gpu);

    if (--count == 0) {
      return gpus;
    }
  }

  UNREACHABLE();
}


Future<Nothing> NvidiaGpuAllocatorProcess::allocate(const set<Gpu>& gpus)
{
  set<Gpu> availableCopy = available;
  set<Gpu> takenCopy = taken;

  foreach (const Gpu& gpu, gpus) {
    size_t count = availableCopy.erase(gpu);
    if (count == 0) {
      return Failure("Error allocating set of GPUs"
                     " from the 'NvidiaGpuAllocator'");
    }
    takenCopy.insert(gpu);
  }

  available = availableCopy;
  taken = takenCopy;

  return Nothing();
}


Future<Nothing> NvidiaGpuAllocatorProcess::deallocate(const set<Gpu>& gpus)
{
  set<Gpu> availableCopy = available;
  set<Gpu> takenCopy = taken;

  foreach (const Gpu& gpu, gpus) {
    size_t count = takenCopy.erase(gpu);
    if (count == 0) {
      return Failure("Error freeing set of GPUs"
                     " back to the 'NvidiaGpuAllocator'");
    }
    availableCopy.insert(gpu);
  }

  available = availableCopy;
  taken = takenCopy;

  return Nothing();
}

} // namespace slave {
} // namespace internal {
} // namespace mesos {
