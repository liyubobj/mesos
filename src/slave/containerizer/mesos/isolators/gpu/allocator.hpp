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

#ifndef __NVIDIA_GPU_ALLOCATOR_HPP__
#define __NVIDIA_GPU_ALLOCATOR_HPP__

#include <set>

#include <mesos/resources.hpp>

#include <process/future.hpp>
#include <process/process.hpp>

#include <stout/option.hpp>
#include <stout/try.hpp>

#include "slave/flags.hpp"

namespace mesos {
namespace internal {
namespace slave {

// Simple abstraction of a GPU.
//
// TODO(klueska): Once we have a generic "Device" type it will look
// very similar to this. At that point we should build redefine this
// abstraction in terms of it.
struct Gpu
{
  unsigned int major;
  unsigned int minor;
};


static inline bool operator<(const Gpu& left, const Gpu& right)
{
  if (left.major < right.major) {
    return true;
  }
  return left.minor < right.minor;
}

static inline bool operator>(const Gpu& left, const Gpu& right)
{
  return left < right;
}

static inline bool operator<=(const Gpu& left, const Gpu& right)
{
  return !(left > right);
}

static inline bool operator>=(const Gpu& left, const Gpu& right)
{
  return !(left < right);
}

static inline bool operator==(const Gpu& left, const Gpu& right)
{
  return left.major == right.major && left.minor == right.minor;
}

static inline bool operator!=(const Gpu& left, const Gpu& right)
{
  return !(left == right);
}


// Forward declaration.
class NvidiaGpuAllocatorProcess;


// The `NvidiaGpuAllocator` class provides an asynchronous method of
// allocating GPUs to multiple libprocess actors. It is overloaded to
// provide proper enumeration of the `gpus` resources from parsing the
// given agent flags.
class NvidiaGpuAllocator
{
public:
  static Try<Resources> resources(const Flags& flags);

  static Try<NvidiaGpuAllocator*> create(const Flags& flags);
  ~NvidiaGpuAllocator();

  const std::set<Gpu>& allGpus() const;

  process::Future<Option<std::set<Gpu>>> allocate(size_t count) const;
  process::Future<Nothing> allocate(const std::set<Gpu>& gpus) const;
  process::Future<Nothing> deallocate(const std::set<Gpu>& gpus) const;

private:
  NvidiaGpuAllocator(const std::set<Gpu>& gpus);
  NvidiaGpuAllocator(const NvidiaGpuAllocator&) = delete;
  void operator=(const NvidiaGpuAllocator&) = delete;

  const std::set<Gpu> gpus;
  process::Owned<NvidiaGpuAllocatorProcess> process;
};


class NvidiaGpuAllocatorProcess
  : public process::Process<NvidiaGpuAllocatorProcess>
{
public:
  NvidiaGpuAllocatorProcess(const std::set<Gpu>& gpus);
  ~NvidiaGpuAllocatorProcess() {}

  process::Future<Option<std::set<Gpu>>> allocate(size_t count);
  process::Future<Nothing> allocate(const std::set<Gpu>& gpus);
  process::Future<Nothing> deallocate(const std::set<Gpu>& gpus);

private:
  std::set<Gpu> available;
  std::set<Gpu> taken;
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif // __NVIDIA_GPU_ALLOCATOR_HPP__
