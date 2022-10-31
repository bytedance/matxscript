// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of Exception is inspired by pythran.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once

#include "functor_str.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

namespace details {
inline std::string JoinSTLStringList(std::initializer_list<std::string> messages) {
  std::string result;
  auto b = messages.begin();
  auto e = messages.end();
  if (b != e) {
    result.append(*b);
    ++b;
  }
  for (; b != e; ++b) {
    // result.append(", ");
    result.append(*b);
  }
  return result;
}

template <class... Types>
inline std::string ToString(Types const&... types) {
  std::initializer_list<std::string> messages{builtins::functor::str(types)...};
  return JoinSTLStringList(messages);
}

inline std::string FormatLineMessage(const char* file,
                                     int line,
                                     const char* cls,
                                     const std::string& what) {
  DateLogger date;
  std::string line_log;
  line_log.append("[").append(date.HumanDate()).append("] ");
  line_log.append(file).append(":").append(std::to_string(line)).append(": ");
  line_log.append(cls).append(": ").append(what);
#if MATXSCRIPT_LOG_STACK_TRACE
  if (ENV_ENABLE_MATX_LOG_STACK_TRACE) {
    line_log.append("\n");
    line_log.append(StackTrace(1, LogStackTraceLevel()));
    line_log.append("\n");
  }
#endif
  return line_log;
}
}  // namespace details

class BaseException : public std::runtime_error {
 public:
  BaseException(const BaseException& e) = default;
  inline BaseException(const std::string& what);
  inline BaseException(const char* file, int line, const std::string& what);
  virtual ~BaseException() noexcept = default;
  virtual const char* ClassName() const {
    return "BaseException";
  }
};

inline BaseException::BaseException(const std::string& what) : std::runtime_error(what) {
}
inline BaseException::BaseException(const char* file, int line, const std::string& what)
    : std::runtime_error(details::FormatLineMessage(file, line, "BaseException", what)) {
}

// Use this to create a python exception class
#define PY_CLASS_EXCEPTION_DECL(Name, Base)                           \
  class Name : public Base {                                          \
   public:                                                            \
    Name(const Name& e) = default;                                    \
    Name(const std::string& what) : Base(what) {                      \
    }                                                                 \
    Name(const char* file, int line, const std::string& what)         \
        : Base(details::FormatLineMessage(file, line, #Name, what)) { \
    }                                                                 \
    const char* ClassName() const override {                          \
      return #Name;                                                   \
    }                                                                 \
    virtual ~Name() noexcept = default;                               \
  }

PY_CLASS_EXCEPTION_DECL(SystemExit, BaseException);
PY_CLASS_EXCEPTION_DECL(KeyboardInterrupt, BaseException);
PY_CLASS_EXCEPTION_DECL(GeneratorExit, BaseException);
PY_CLASS_EXCEPTION_DECL(Exception, BaseException);
PY_CLASS_EXCEPTION_DECL(StopIteration, Exception);
PY_CLASS_EXCEPTION_DECL(StandardError, Exception);
PY_CLASS_EXCEPTION_DECL(Warning, Exception);
PY_CLASS_EXCEPTION_DECL(BytesWarning, Warning);
PY_CLASS_EXCEPTION_DECL(UnicodeWarning, Warning);
PY_CLASS_EXCEPTION_DECL(ImportWarning, Warning);
PY_CLASS_EXCEPTION_DECL(FutureWarning, Warning);
PY_CLASS_EXCEPTION_DECL(UserWarning, Warning);
PY_CLASS_EXCEPTION_DECL(SyntaxWarning, Warning);
PY_CLASS_EXCEPTION_DECL(RuntimeWarning, Warning);
PY_CLASS_EXCEPTION_DECL(PendingDeprecationWarning, Warning);
PY_CLASS_EXCEPTION_DECL(DeprecationWarning, Warning);
PY_CLASS_EXCEPTION_DECL(BufferError, StandardError);
PY_CLASS_EXCEPTION_DECL(FileNotFoundError, StandardError);
PY_CLASS_EXCEPTION_DECL(ArithmeticError, StandardError);
PY_CLASS_EXCEPTION_DECL(AssertionError, StandardError);
PY_CLASS_EXCEPTION_DECL(AttributeError, StandardError);
PY_CLASS_EXCEPTION_DECL(EnvironmentError, StandardError);
PY_CLASS_EXCEPTION_DECL(EOFError, StandardError);
PY_CLASS_EXCEPTION_DECL(ImportError, StandardError);
PY_CLASS_EXCEPTION_DECL(LookupError, StandardError);
PY_CLASS_EXCEPTION_DECL(MemoryError, StandardError);
PY_CLASS_EXCEPTION_DECL(NameError, StandardError);
PY_CLASS_EXCEPTION_DECL(ReferenceError, StandardError);
PY_CLASS_EXCEPTION_DECL(RuntimeError, StandardError);
PY_CLASS_EXCEPTION_DECL(SyntaxError, StandardError);
PY_CLASS_EXCEPTION_DECL(SystemError, StandardError);
PY_CLASS_EXCEPTION_DECL(TypeError, StandardError);
PY_CLASS_EXCEPTION_DECL(ValueError, StandardError);
PY_CLASS_EXCEPTION_DECL(FloatingPointError, ArithmeticError);
PY_CLASS_EXCEPTION_DECL(OverflowError, ArithmeticError);
PY_CLASS_EXCEPTION_DECL(ZeroDivisionError, ArithmeticError);
PY_CLASS_EXCEPTION_DECL(IOError, EnvironmentError);
PY_CLASS_EXCEPTION_DECL(OSError, EnvironmentError);
PY_CLASS_EXCEPTION_DECL(WindowsError, OSError);
PY_CLASS_EXCEPTION_DECL(VMSError, OSError);
PY_CLASS_EXCEPTION_DECL(IndexError, LookupError);
PY_CLASS_EXCEPTION_DECL(KeyError, LookupError);
PY_CLASS_EXCEPTION_DECL(UnboundLocalError, NameError);
PY_CLASS_EXCEPTION_DECL(NotImplementedError, RuntimeError);
PY_CLASS_EXCEPTION_DECL(IndentationError, SyntaxError);
PY_CLASS_EXCEPTION_DECL(TabError, IndentationError);
PY_CLASS_EXCEPTION_DECL(UnicodeError, ValueError);

#define __MAKE_PY_EXCEPTION__(_EX_CLS, ...) \
  ::matxscript::runtime::_EX_CLS(__FILE__, __LINE__, details::ToString(__VA_ARGS__))

#define __THROW_PY_EXCEPTION__(...) throw __MAKE_PY_EXCEPTION__(__VA_ARGS__)

// MAKE_PY_XX is for raise codegen

#define MAKE_PY_Exception(...) __MAKE_PY_EXCEPTION__(Exception, __VA_ARGS__)
#define THROW_PY_Exception(...) __THROW_PY_EXCEPTION__(Exception, __VA_ARGS__)

#define MAKE_PY_BaseException(...) __MAKE_PY_EXCEPTION__(BaseException, __VA_ARGS__)
#define THROW_PY_BaseException(...) __THROW_PY_EXCEPTION__(BaseException, __VA_ARGS__)

#define MAKE_PY_StopIteration(...) __MAKE_PY_EXCEPTION__(StopIteration, __VA_ARGS__)
#define THROW_PY_StopIteration(...) __THROW_PY_EXCEPTION__(StopIteration, __VA_ARGS__)

#define MAKE_PY_ValueError(...) __MAKE_PY_EXCEPTION__(ValueError, __VA_ARGS__)
#define THROW_PY_ValueError(...) __THROW_PY_EXCEPTION__(ValueError, __VA_ARGS__)

#define MAKE_PY_OverflowError(...) __MAKE_PY_EXCEPTION__(OverflowError, __VA_ARGS__)
#define THROW_PY_OverflowError(...) __THROW_PY_EXCEPTION__(OverflowError, __VA_ARGS__)

#define MAKE_PY_TypeError(...) __MAKE_PY_EXCEPTION__(TypeError, __VA_ARGS__)
#define THROW_PY_TypeError(...) __THROW_PY_EXCEPTION__(TypeError, __VA_ARGS__)

#define MAKE_PY_OSError(...) __MAKE_PY_EXCEPTION__(OSError, __VA_ARGS__)
#define THROW_PY_OSError(...) __THROW_PY_EXCEPTION__(OSError, __VA_ARGS__)

#define MAKE_PY_RuntimeError(...) __MAKE_PY_EXCEPTION__(RuntimeError, __VA_ARGS__)
#define THROW_PY_RuntimeError(...) __THROW_PY_EXCEPTION__(RuntimeError, __VA_ARGS__)

#define MAKE_PY_SystemError(...) __MAKE_PY_EXCEPTION__(SystemError, __VA_ARGS__)
#define THROW_PY_SystemError(...) __THROW_PY_EXCEPTION__(SystemError, __VA_ARGS__)

#define MAKE_PY_MemoryError(...) __MAKE_PY_EXCEPTION__(MemoryError, __VA_ARGS__)
#define THROW_PY_MemoryError(...) __THROW_PY_EXCEPTION__(MemoryError, __VA_ARGS__)

#define MAKE_PY_ZeroDivisionError(...) __MAKE_PY_EXCEPTION__(ZeroDivisionError, __VA_ARGS__)
#define THROW_PY_ZeroDivisionError(...) __THROW_PY_EXCEPTION__(ZeroDivisionError, __VA_ARGS__)

#define MAKE_PY_IndexError(...) __MAKE_PY_EXCEPTION__(IndexError, __VA_ARGS__)
#define THROW_PY_IndexError(...) __THROW_PY_EXCEPTION__(IndexError, __VA_ARGS__)

#define MAKE_PY_AttributeError(...) __MAKE_PY_EXCEPTION__(AttributeError, __VA_ARGS__)
#define THROW_PY_AttributeError(...) __THROW_PY_EXCEPTION__(AttributeError, __VA_ARGS__)

#define MAKE_PY_AssertionError(...) __MAKE_PY_EXCEPTION__(AssertionError, __VA_ARGS__)
#define THROW_PY_AssertionError(...) __THROW_PY_EXCEPTION__(AssertionError, __VA_ARGS__)

#define MAKE_PY_NotImplementedError(...) __MAKE_PY_EXCEPTION__(NotImplementedError, __VA_ARGS__)
#define THROW_PY_NotImplementedError(...) __THROW_PY_EXCEPTION__(NotImplementedError, __VA_ARGS__)

}  // namespace runtime
}  // namespace matxscript
