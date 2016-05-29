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
#include <string>
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
using std::string;
using std::vector;

namespace mesos {
namespace internal {
namespace slave {

static constexpr unsigned int NVIDIA_MAJOR_DEVICE = 195;


// To determine the proper number of GPU resources to return, we
// need to check both --resources and --nvidia_gpu_devices.
// There are two cases to consider:
//
//   (1) --resources includes "gpus" and --nvidia_gpu_devices is set.
//       The number of GPUs in --resources must equal the number of
//       GPUs within --nvidia_gpu_resources.
//
//   (2) --resources does not include "gpus" and --nvidia_gpu_devices
//       is not specified. Here we auto-discover GPUs using the
//       NVIDIA management Library (NVML). We special case specifying
//       `gpus:0` explicitly to not perform auto-discovery.
//
// NOTE: We also check to make sure the `gpu/nvidia` isolation flag
// is set before enumerating GPUs. We do this because we decided it
// makes sense to only do autodiscovery of GPUs when this isolator
// is turned on (unlike for CPUs, memory, and disk where
// autodiscovery happens by default). We decided to take this
// approach, because GPU support is still experimental, and is only
// known to work well if this isolator is enabled. We didn't want to
// start advertising GPUs in our resource offer and have people
// attempt to use them in scenarious we haven't considered yet. In
// the future we may support other use cases, but for now we are
// being cautious.
static Try<Resources> enumerateResources(const Flags& flags)
{
  const vector<string> tokens = strings::tokenize(flags.isolation, ",");
  const set<string> isolators = set<string>(tokens.begin(), tokens.end());

  // Don't allow the `--nvidia-gpu_devices` flag without the GPU isolator.
  if (flags.nvidia_gpu_devices.isSome() && isolators.count("gpu/nvidia") == 0) {
    return Error("'--nvidia_gpus_devices' can only be specified if the"
                 " `--isolation` flag contains 'gpu/nvidia'");
  }

  // Pull out just the GPU resources from --resources.
  Try<Resources> parsed = Resources::parse(
      flags.resources.getOrElse(""), flags.default_role);

  if (parsed.isError()) {
    return Error(parsed.error());
  }

  Resources resources = parsed.get().filter(
      [](const Resource& resource) {
          return resource.name() == "gpus";
      });

  // Pass the GPU resources through.
  if (isolators.count("gpu/nvidia") == 0 || !nvml::isAvailable()) {
    return resources;
  }

  // Enumerate GPUs based on the flags.
  Try<Nothing> initialized = nvml::initialize();
  if (initialized.isError()) {
    return Error("Failed to nvml::initialize: " + initialized.error());
  }

  Try<unsigned int> available = nvml::deviceGetCount();
  if (available.isError()) {
    return Error("Failed to nvml::deviceGetCount: " + available.error());
  }

  // Using the resource vector alone, there is no way to distinguish
  // between a user specifying `gpus:0` in the `--resources` flag and
  // not specifying `gpus` at all. To help with this we short circuit
  // this function to return an empty resource vector for the case of
  // explicitly setting `gpus:0'. By doing so, it is sufficient in the
  // rest of this function to call `resources.gpus().isSome()` to
  // determine if a value was explicitly set for `gpus` or not.
  if (strings::contains(flags.resources.getOrElse(""), "gpus") &&
      !resources.gpus().isSome()) {
    if (flags.nvidia_gpu_devices.isSome()) {
      return Error("'--nvidia_gpus_devices' cannot be specified"
                   " when '--resources' specifies 0 GPUs");
    }
    return Resources();
  }

  if (flags.nvidia_gpu_devices.isSome() && !resources.gpus().isSome()) {
    return Error("'--nvidia_gpus_devices' cannot be set without"
                 " also setting 'gpus' in '--resources'");
  }

  if (resources.gpus().isSome() && !flags.nvidia_gpu_devices.isSome()) {
    return Error("The `gpus` resource can not be set without also"
                 " setting `--nvidia_gpu_devices`");
  }

  if (resources.gpus().isSome()) {
    // Make sure that the value of "gpus" is an integer and not a
    // fractional amount. We take advantage of the fact that we know
    // the value of "gpus" is only precise up to 3 decimals.
    long long milli = static_cast<long long>(resources.gpus().get() * 1000);
    if ((milli % 1000) != 0) {
      return Error("The 'gpus' resource must be an non-negative integer");
    }

    // Make sure the `nvidia_gpu_devices` flag
    // contains a list of unique GPU identifiers.
    vector<unsigned int> unique = flags.nvidia_gpu_devices.get();
    std::sort(unique.begin(), unique.end());
    auto last = std::unique(unique.begin(), unique.end());
    unique.erase(last, unique.end());

    if (unique.size() != flags.nvidia_gpu_devices->size()) {
      return Error("'--nvidia_gpu_devices' contains duplicates");
    }

    if (flags.nvidia_gpu_devices->size() != resources.gpus().get()) {
      return Error("'--resources' and '--nvidia_gpu_devices' specify"
                   " different numbers of GPU devices");
    }

    if (resources.gpus().get() > available.get()) {
      return Error("The number of GPUs requested is greater than"
                   " the number of GPUs available on the machine");
    }

    return resources;
  }

  return Resources::parse(
      "gpus",
      stringify(available.get()),
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
