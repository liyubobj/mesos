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

#ifndef __NVIDIA_NVML_HPP__
#define __NVIDIA_NVML_HPP__

#include <nvidia/gdk/nvml.h>

#include <stout/try.hpp>

// The `nvml` abstraction serves to load Nvidia's `libnvidia-ml`
// library as a runtime dependency and make a subset of its functions
// available for general use. Making this library a run-time
// dependence (instead of a link-time dependence) allows us to follow
// different code paths depending on whether the library is actually
// available on the system or not. This is advantageous for deploying
// a common `libmesos` on machines both with and without GPUs
// installed. Only those machines that actually have GPUs need to have
// the `libnvidia-ml` library installed on their system. Master nodes
// and agents without GPUs do not.
namespace nvml {

// Check if the NVML library is installed on the system as is able to
// be loaded as a runtime dependence.
bool isAvailable();

// Initialize NVML for use. If this call fails, none of the other
// functions in this abstraction will succeed.
Try<Nothing> initialize();

// Finalize the use of the NVML abstraction. This call will only
// succeed after a previously successful call to `initialize()`.
// After successfully calling `finalize()` the NVML abstraction can be
// reinitialized by a subsequent call to `initialize()`.
Try<Nothing> finalize();

// NVML specific functions. These calls will only success after a
// previously successful call to `initialize()`.
Try<unsigned int> deviceGetCount();
Try<nvmlDevice_t> deviceGetHandleByIndex(unsigned int index);
Try<unsigned int> deviceGetMinorNumber(nvmlDevice_t handle);

} // namespace nvml {

#endif // __NVIDIA_NVML_HPP__
