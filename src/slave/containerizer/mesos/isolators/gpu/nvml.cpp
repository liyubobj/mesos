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

#include <dlfcn.h>

#include <nvidia/gdk/nvml.h>

#include <string>

#include <process/once.hpp>

#include <stout/exit.hpp>
#include <stout/nothing.hpp>
#include <stout/stringify.hpp>
#include <stout/try.hpp>

#include <stout/posix/dynamiclibrary.hpp>

#include "slave/containerizer/mesos/isolators/gpu/nvml.hpp"

using process::Once;

namespace nvml {

constexpr char LIBRARY_NAME[] = "libnvidia-ml.so.1";


struct NvidiaManagementLibrary
{
  NvidiaManagementLibrary(
      nvmlReturn_t (*_deviceGetCount)(unsigned int*),
      nvmlReturn_t (*_deviceGetHandleByIndex)(unsigned int, nvmlDevice_t*),
      nvmlReturn_t (*_deviceGetMinorNumber)(nvmlDevice_t, unsigned int*),
      const char* (*_errorString)(nvmlReturn_t))
    : deviceGetCount(_deviceGetCount),
      deviceGetHandleByIndex(_deviceGetHandleByIndex),
      deviceGetMinorNumber(_deviceGetMinorNumber),
      errorString(_errorString) {}

  nvmlReturn_t (*deviceGetCount)(unsigned int*);
  nvmlReturn_t (*deviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
  nvmlReturn_t (*deviceGetMinorNumber)(nvmlDevice_t, unsigned int*);
  const char* (*errorString)(nvmlReturn_t);
};


const NvidiaManagementLibrary* nvml = nullptr;


Once* initialized = new Once();
Option<Error>* error = new Option<Error>();
DynamicLibrary* library = new DynamicLibrary();


Try<Nothing> initialize()
{
  if (initialized->once()) {
    if (error->isSome()) {
      return error->get();
    }
    return Nothing();
  }

  Try<Nothing> open = library->open(LIBRARY_NAME);
  if (open.isError()) {
    *error = open.error();
    initialized->done();
    return error->get();
  }

  Try<void*> symbol = library->loadSymbol("nvmlInit_v2");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlInit = (nvmlReturn_t (*)())symbol.get();

  symbol = library->loadSymbol("nvmlDeviceGetCount");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlDeviceGetCount = (nvmlReturn_t (*)(unsigned int*))symbol.get();

  symbol = library->loadSymbol("nvmlDeviceGetHandleByIndex");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlDeviceGetHandleByIndex =
    (nvmlReturn_t (*)(unsigned int, nvmlDevice_t*))symbol.get();

  symbol = library->loadSymbol("nvmlDeviceGetMinorNumber");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlDeviceGetMinorNumber =
    (nvmlReturn_t (*)(nvmlDevice_t, unsigned int*))symbol.get();

  symbol = library->loadSymbol("nvmlErrorString");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlErrorString = (const char* (*)(nvmlReturn_t))symbol.get();

  nvmlReturn_t result = nvmlInit();
  if (result != NVML_SUCCESS) {
    *error = Error("nvmlInit failed: " +  stringify(nvmlErrorString(result)));
    initialized->done();
    return error->get();
  }

  nvml = new NvidiaManagementLibrary(
      nvmlDeviceGetCount,
      nvmlDeviceGetHandleByIndex,
      nvmlDeviceGetMinorNumber,
      nvmlErrorString);

  initialized->done();

  return Nothing();
}


Try<Nothing> finalize()
{
  if (nvml == nullptr) {
    return Error("NVML has not been initialized");
  }

  delete nvml;
  delete library;
  delete error;
  delete initialized;

  return Nothing();
}


bool isAvailable()
{
  // Unfortunately, there is no function available in `glibc` to check
  // if a dynamic library is available to open with `dlopen()`.
  // Instead, availablity is determined by attempting to open a
  // library with `dlopen()` and if this call fails, assuming the
  // library is unavailable. The problem with using this method to
  // implement a generic `isAvailable()` function is knowing if we
  // should call `dlclose()` on the library once we've determined that
  // the library is in fact available (because some other code path
  // may have already opened the library and we don't want to close it
  // out from under them). Luckily calls to `dlopen()` are reference
  // counted, so that subsequent calls to `dlclose()` simply down the
  // reference count and only actually close the library when this
  // reference count hits zero. Because of this, we can
  // unconditionally call `dlclose()` and trust that glibc will take
  // care to do the right thing. Additionally, calling `dlopen()` with
  // `RTLD_LAZY` is the preferred method here because it is faster in
  // cases where the library is not yet opened, and having previously
  // opened it with `RTLD_NOW` will always take precedence.
  void* open = ::dlopen(LIBRARY_NAME, RTLD_LAZY);
  if (open == NULL) {
    return false;
  }

  CHECK_EQ(0, ::dlclose(open))
    << "dlcose failed: " << dlerror();

  return true;
}


Try<unsigned int> deviceGetCount()
{
  if (nvml == nullptr) {
    return Error("NVML has not been initialized");
  }

  unsigned int count;
  nvmlReturn_t result = nvml->deviceGetCount(&count);
  if (result != NVML_SUCCESS) {
    return Error(stringify(nvml->errorString(result)));
  }
  return count;
}


Try<nvmlDevice_t> deviceGetHandleByIndex(unsigned int index)
{
  if (nvml == nullptr) {
    return Error("NVML has not been initialized");
  }

  nvmlDevice_t handle;
  nvmlReturn_t result = nvml->deviceGetHandleByIndex(index, &handle);
  if (result == NVML_ERROR_INVALID_ARGUMENT) {
    return Error("GPU device " + stringify(index) + " not found");
  }
  if (result != NVML_SUCCESS) {
    return Error(stringify(nvml->errorString(result)));
  }
  return handle;
}


Try<unsigned int> deviceGetMinorNumber(nvmlDevice_t handle)
{
  if (nvml == nullptr) {
    return Error("NVML has not been initialized");
  }

  unsigned int minor;
  nvmlReturn_t result = nvml->deviceGetMinorNumber(handle, &minor);
  if (result != NVML_SUCCESS) {
    return Error(stringify(nvml->errorString(result)));
  }
  return minor;
}

} // namespace nvml {
