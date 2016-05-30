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

#include <stdint.h>

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <process/collect.hpp>
#include <process/defer.hpp>
#include <process/future.hpp>
#include <process/shared.hpp>

#include <stout/error.hpp>
#include <stout/foreach.hpp>
#include <stout/hashmap.hpp>
#include <stout/option.hpp>
#include <stout/os.hpp>
#include <stout/try.hpp>

#include "linux/cgroups.hpp"

#include "slave/flags.hpp"

#include "slave/containerizer/containerizer.hpp"

#include "slave/containerizer/mesos/isolator.hpp"

#include "slave/containerizer/mesos/isolators/gpu/nvidia.hpp"
#include "slave/containerizer/mesos/isolators/gpu/nvml.hpp"

using cgroups::devices::Entry;

using mesos::slave::ContainerConfig;
using mesos::slave::ContainerLaunchInfo;
using mesos::slave::ContainerLimitation;
using mesos::slave::ContainerState;
using mesos::slave::Isolator;

using mesos::internal::slave::Containerizer;

using process::defer;
using process::Failure;
using process::Future;
using process::PID;
using process::Shared;

using std::list;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace mesos {
namespace internal {
namespace slave {

// TODO(klueska): Expand this when we support other GPU types.
static constexpr unsigned int NVIDIA_MAJOR_DEVICE = 195;


NvidiaGpuIsolatorProcess::NvidiaGpuIsolatorProcess(
    const Flags& _flags,
    const string& _hierarchy,
    const Shared<NvidiaGpuAllocator>& _allocator,
    const cgroups::devices::Entry& uvmDeviceEntry,
    const cgroups::devices::Entry& ctlDeviceEntry)
  : flags(_flags),
    hierarchy(_hierarchy),
    allocator(_allocator),
    NVIDIA_CTL_DEVICE_ENTRY(ctlDeviceEntry),
    NVIDIA_UVM_DEVICE_ENTRY(uvmDeviceEntry) {}


Try<Isolator*> NvidiaGpuIsolatorProcess::create(const Flags& flags)
{
  // Make sure the `cgroups/devices` isolator is present.
  vector<string> isolators_ = strings::tokenize(flags.isolation, ",");
  set<string> isolators = set<string>(isolators_.begin(), isolators_.end());
  if (isolators.count("cgroups/devices") == 0) {
    return Error("The 'cgroups/devices' isolator must be enabled in"
                 " order to use the gpu/devices isolator");
  }

  // Create an `NvidiaGpuAllocator` instance. Eventually this
  // should be moved above this component so that it can be
  // shared with the docker containerizer.
  Try<NvidiaGpuAllocator*> _allocator = NvidiaGpuAllocator::create(flags);
  if (_allocator.isError()) {
    return Error(_allocator.error());
  }

  Shared<NvidiaGpuAllocator> allocator(_allocator.get());

  // Populate the device entries for
  // `/dev/nvidiactl` and `/dev/nvidia-uvm`.
  Try<dev_t> device = os::stat::rdev("/dev/nvidiactl");
  if (device.isError()) {
    return Error(device.error());
  }
  cgroups::devices::Entry ctlDeviceEntry;
  ctlDeviceEntry.selector.type = Entry::Selector::Type::CHARACTER;
  ctlDeviceEntry.selector.major = major(device.get());
  ctlDeviceEntry.selector.minor = minor(device.get());
  ctlDeviceEntry.access.read = true;
  ctlDeviceEntry.access.write = true;
  ctlDeviceEntry.access.mknod = true;

  device = os::stat::rdev("/dev/nvidia-uvm");
  if (device.isError()) {
    return Error(device.error());
  }
  cgroups::devices::Entry uvmDeviceEntry;
  uvmDeviceEntry.selector.type = Entry::Selector::Type::CHARACTER;
  uvmDeviceEntry.selector.major = major(device.get());
  uvmDeviceEntry.selector.minor = minor(device.get());
  uvmDeviceEntry.access.read = true;
  uvmDeviceEntry.access.write = true;
  uvmDeviceEntry.access.mknod = true;

  // Retrieve the cgroups devices hierarchy.
  Result<string> hierarchy = cgroups::hierarchy("devices");

  if (hierarchy.isError()) {
    return Error(
        "Error retrieving the 'devices' subsystem hierarchy: " +
        hierarchy.error());
  }

  process::Owned<MesosIsolatorProcess> process(
      new NvidiaGpuIsolatorProcess(
          flags,
          hierarchy.get(),
          allocator,
          ctlDeviceEntry,
          uvmDeviceEntry));

  return new MesosIsolator(process);
}


Future<Nothing> NvidiaGpuIsolatorProcess::recover(
    const list<ContainerState>& states,
    const hashset<ContainerID>& orphans)
{
  list<Future<Nothing>> futures;

  foreach (const ContainerState& state, states) {
    const ContainerID& containerId = state.container_id();
    const string cgroup = path::join(flags.cgroups_root, containerId.value());

    Try<bool> exists = cgroups::exists(hierarchy, cgroup);
    if (exists.isError()) {
      foreachvalue (Info* info, infos) {
        delete info;
      }
      infos.clear();
      return Failure("Failed to check cgroup for container '" +
                     stringify(containerId) + "'");
    }

    if (!exists.get()) {
      VLOG(1) << "Couldn't find cgroup for container " << containerId;
      // This may occur if the executor has exited and the isolator
      // has destroyed the cgroup but the slave dies before noticing
      // this. This will be detected when the containerizer tries to
      // monitor the executor's pid.
      continue;
    }

    infos[containerId] = new Info(containerId, cgroup);

    // Determine which GPUs are allocated to this container.
    Try<vector<cgroups::devices::Entry>> entries =
      cgroups::devices::list(hierarchy, cgroup);

    if (entries.isError()) {
      return Failure("Failed to obtain devices list for cgroup"
                     " '" + cgroup + "': " + entries.error());
    }

    const set<Gpu> allGpus = allocator->allGpus();

    set<Gpu> containerGpus;
    foreach (const cgroups::devices::Entry& entry, entries.get()) {
      for (auto gpu = allGpus.begin(); gpu != allGpus.end(); ++gpu) {
        if (entry.selector.major == gpu->major &&
            entry.selector.minor == gpu->minor) {
          containerGpus.insert(*gpu);
          break;
        }
      }
    }

    futures.push_back(allocator->allocate(containerGpus)
      .then(defer(self(), [=]() -> Future<Nothing> {
        infos[containerId]->allocated = containerGpus;
        return Nothing();
      })));
  }

  return collect(futures).then([]() { return Nothing(); });
}


Future<Option<ContainerLaunchInfo>> NvidiaGpuIsolatorProcess::prepare(
    const ContainerID& containerId,
    const mesos::slave::ContainerConfig& containerConfig)
{
  if (infos.contains(containerId)) {
    return Failure("Container has already been prepared");
  }

  infos[containerId] = new Info(
      containerId, path::join(flags.cgroups_root, containerId.value()));

  return update(containerId, containerConfig.executor_info().resources())
    .then([]() -> Future<Option<ContainerLaunchInfo>> {
      return None();
    });
}


Future<Nothing> NvidiaGpuIsolatorProcess::update(
    const ContainerID& containerId,
    const Resources& resources)
{
  if (!infos.contains(containerId)) {
    return Failure("Unknown container");
  }

  Info* info = CHECK_NOTNULL(infos[containerId]);

  Option<double> gpus = resources.gpus();

  // Make sure that the `gpus` resource is not fractional.
  // We rely on scalar resources only having 3 digits of precision.
  if (static_cast<long long>(gpus.getOrElse(0.0) * 1000.0) % 1000 != 0) {
    return Failure("The 'gpus' resource must be an unsigned integer");
  }

  size_t requested = static_cast<size_t>(resources.gpus().getOrElse(0.0));

  // Update the GPU allocation to reflect the new total.
  if (requested > info->allocated.size()) {
    size_t additional = requested - info->allocated.size();

    return allocator->allocate(additional)
      .then(defer(self(), [=](Option<set<Gpu>> allocated) -> Future<Nothing> {
        if (!allocated.isSome()) {
          return Failure("Not enough GPUs available to reserve"
                         " " + stringify(additional) + " additional GPUs");
        }

        // Grant access to /dev/nvidiactl and /dev/nvida-uvm
        // if this container is about to get its first GPU.
        if (info->allocated.empty()) {
          map<string, const cgroups::devices::Entry> entries = {
            { "/dev/nvidiactl", NVIDIA_CTL_DEVICE_ENTRY },
            { "/dev/nvidia-uvm", NVIDIA_UVM_DEVICE_ENTRY },
          };

          foreachkey (const string& device, entries) {
            Try<Nothing> allow = cgroups::devices::allow(
                hierarchy, info->cgroup, entries[device]);

            if (allow.isError()) {
              return Failure("Failed to grant cgroups access to"
                             " '" + device + "': " + allow.error());
            }
          }
        }

        foreach (const Gpu& gpu, allocated.get()) {
          cgroups::devices::Entry entry;
          entry.selector.type = Entry::Selector::Type::CHARACTER;
          entry.selector.major = gpu.major;
          entry.selector.minor = gpu.minor;
          entry.access.read = true;
          entry.access.write = true;
          entry.access.mknod = true;

          Try<Nothing> allow = cgroups::devices::allow(
              hierarchy, info->cgroup, entry);

          if (allow.isError()) {
            return Failure("Failed to grant cgroups access to GPU device"
                           " '" + stringify(entry) + "': " + allow.error());
          }
        }

        info->allocated = allocated.get();

        return Nothing();
      }));
  } else if (requested < info->allocated.size()) {
    size_t fewer = info->allocated.size() - requested;

    set<Gpu> deallocated;

    for (size_t i = 0; i < fewer; i++) {
      const auto gpu = info->allocated.begin();

      cgroups::devices::Entry entry;
      entry.selector.type = Entry::Selector::Type::CHARACTER;
      entry.selector.major = gpu->major;
      entry.selector.minor = gpu->minor;
      entry.access.read = true;
      entry.access.write = true;
      entry.access.mknod = true;

      Try<Nothing> deny = cgroups::devices::deny(
          hierarchy, info->cgroup, entry);

      if (deny.isError()) {
        return Failure("Failed to deny cgroups access to GPU device"
                       " '" + stringify(entry) + "': " + deny.error());
      }

      deallocated.insert(*gpu);
      info->allocated.erase(gpu);
    }

    return allocator->deallocate(deallocated)
      .then([=] () -> Future<Nothing> {
        // Revoke access from /dev/nvidiactl and /dev/nvida-uvm
        // if this container no longer has access to any GPUs.
        if (info->allocated.empty()) {
          map<string, const cgroups::devices::Entry> entries = {
            { "/dev/nvidiactl", NVIDIA_CTL_DEVICE_ENTRY },
            { "/dev/nvidia-uvm", NVIDIA_UVM_DEVICE_ENTRY },
          };

          foreachkey (const string& device, entries) {
            Try<Nothing> deny = cgroups::devices::deny(
                hierarchy, info->cgroup, entries[device]);

            if (deny.isError()) {
              return Failure("Failed to deny cgroups access to"
                             " '" + device + "': " + deny.error());
            }
          }
        }

        return Nothing();
      });
  }

  return Nothing();
}


Future<ResourceStatistics> NvidiaGpuIsolatorProcess::usage(
    const ContainerID& containerId)
{
  if (!infos.contains(containerId)) {
    return Failure("Unknown container");
  }

  // TODO(rtodd): Obtain usage information from NVML.

  ResourceStatistics result;
  return result;
}


Future<Nothing> NvidiaGpuIsolatorProcess::cleanup(
    const ContainerID& containerId)
{
  // Multiple calls may occur during test clean up.
  if (!infos.contains(containerId)) {
    VLOG(1) << "Ignoring cleanup request for unknown container " << containerId;

    return Nothing();
  }

  Info* info = CHECK_NOTNULL(infos[containerId]);

  // Make any remaining GPUs available.
  return allocator->deallocate(info->allocated)
    .onFailed([=] (const string& message) -> Future<Nothing> {
      return Failure(
          "Failed to clean up container " +
          stringify(containerId) + ": " + message);
    })
    .then(defer([=]() -> Future<Nothing> {
      delete info;
      infos.erase(containerId);

      return Nothing();
    }));
}

} // namespace slave {
} // namespace internal {
} // namespace mesos {
