// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __STOUT_ERROR_HPP__
#define __STOUT_ERROR_HPP__

#include <errno.h>

#include <string>

#include <stout/os/strerror.hpp>

// A useful type that can be used to represent a Try that has
// failed. You can also use 'ErrnoError' to append the error message
// associated with the current 'errno' to your own error message.
//
// Examples:
//
//   Result<int> result = Error("uninitialized");
//   Try<std::string> = Error("uninitialized");
//
//   void foo(Try<std::string> t) {}
//
//   foo(Error("some error here"));

class Error
{
public:
  explicit Error(const std::string& _message) : message(_message) {}

  const std::string message;
};


class ErrnoError : public Error
{
public:
  ErrnoError() : Error(os::strerror(errno)) {}

  ErrnoError(const std::string& message)
    : Error(message + ": " + os::strerror(errno)) {}
};

#endif // __STOUT_ERROR_HPP__
