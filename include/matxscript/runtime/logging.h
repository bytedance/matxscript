// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
 *
 * Copyright (c) 2015 by Contributors
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

#include "runtime_port.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/* default logic for stack trace */
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__sun) && !defined(__SVR4) && \
     !(defined __MINGW64__) && !(defined __ANDROID__)) &&                                 \
    !defined(__CYGWIN__) && !defined(__EMSCRIPTEN__) && !defined(__RISCV__) &&            \
    !defined(__hexagon__)
#ifndef MATXSCRIPT_LOG_STACK_TRACE
#define MATXSCRIPT_LOG_STACK_TRACE 1
#endif
#ifndef MATXSCRIPT_LOG_STACK_TRACE_SIZE
#define MATXSCRIPT_LOG_STACK_TRACE_SIZE 10
#endif
#endif

namespace matxscript {
namespace runtime {

extern bool ENV_ENABLE_MATX_LOG_STACK_TRACE;

#ifdef MATXSCRIPT_LOG_STACK_TRACE
// get stack trace logging depth from env variable.
inline size_t LogStackTraceLevel() {
  size_t level;
  if (auto var = std::getenv("MATXSCRIPT_LOG_STACK_TRACE_DEPTH")) {
    if (1 == sscanf(var, "%zu", &level)) {
      return level + 1;
    }
  }
  return MATXSCRIPT_LOG_STACK_TRACE_SIZE;
}

// By default skip the first frame because
// that belongs to ~LogMessageFatal
std::string StackTrace(size_t start_frame = 1,
                       const size_t stack_size = MATXSCRIPT_LOG_STACK_TRACE_SIZE);

#else
inline size_t LogStackTraceLevel() {
  return 0;
}

inline std::string StackTrace(size_t start_frame = 1, const size_t stack_size = 0) {
  return std::string("Stack trace not available");
}

#endif

/*!
 * \brief exception class that will be thrown by
 *  default logger if MATXSCRIPT_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string& s) : std::runtime_error(s) {
  }
};

}  // namespace runtime
}  // namespace matxscript

// use a light version of glog
#include <assert.h>
#include <ctime>
#include <iostream>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#pragma warning(disable : 4068)
#endif

namespace matxscript {
namespace runtime {

#define MATXSCRIPT_CHECK_BINARY_OP(name, op, x, y)                    \
  if (!((x)op(y)))                                                    \
  ::matxscript::runtime::LogMessageFatal(__FILE__, __LINE__).stream() \
      << "Check failed: " << #x " " #op " " #y << " (" << (x) << " vs. " << (y) << "): "

// Always-on checking
#define MXCHECK(x) \
  if (!(x))        \
  ::matxscript::runtime::LogMessageFatal(__FILE__, __LINE__).stream() << "Check failed: " #x << ": "
#define MXTHROW ::matxscript::runtime::LogMessageFatal(__FILE__, __LINE__).stream() << ": "
#define MXCHECK_LT(x, y) MATXSCRIPT_CHECK_BINARY_OP(_LT, <, x, y)
#define MXCHECK_GT(x, y) MATXSCRIPT_CHECK_BINARY_OP(_GT, >, x, y)
#define MXCHECK_LE(x, y) MATXSCRIPT_CHECK_BINARY_OP(_LE, <=, x, y)
#define MXCHECK_GE(x, y) MATXSCRIPT_CHECK_BINARY_OP(_GE, >=, x, y)
#define MXCHECK_EQ(x, y) MATXSCRIPT_CHECK_BINARY_OP(_EQ, ==, x, y)
#define MXCHECK_NE(x, y) MATXSCRIPT_CHECK_BINARY_OP(_NE, !=, x, y)
#define MXCHECK_NOTNULL(x)                                                           \
  ((x) == NULL ? ::matxscript::runtime::LogMessageFatal(__FILE__, __LINE__).stream() \
                     << "Check  notnull: " #x << ' ',                                \
   (x)                                                                               \
               : (x))

#define MXLOG_DEBUG \
  ::matxscript::runtime::LogMessage(__FILE__, __LINE__, ::matxscript::runtime::LoggingLevel::DEBUG)
#define MXLOG_INFO \
  ::matxscript::runtime::LogMessage(__FILE__, __LINE__, ::matxscript::runtime::LoggingLevel::INFO)
#define MXLOG_ERROR \
  ::matxscript::runtime::LogMessage(__FILE__, __LINE__, ::matxscript::runtime::LoggingLevel::ERROR)
#define MXLOG_WARNING                \
  ::matxscript::runtime::LogMessage( \
      __FILE__, __LINE__, ::matxscript::runtime::LoggingLevel::WARNING)
#define MXLOG_FATAL ::matxscript::runtime::LogMessageFatal(__FILE__, __LINE__)
#define MXLOG_QFATAL MXLOG_FATAL

#define MXLOG(severity) MXLOG_##severity.stream()
#define MXLG MXLOG_INFO.stream()
#define MXLOG_IF(severity, condition) \
  !(condition) ? (void)0 : ::matxscript::runtime::LogMessageVoidify() & MXLOG(severity)

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char* HumanDate() {
#if !defined(_LIBCPP_SGX_CONFIG) && MATXSCRIPT_LOG_NODATE == 0
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm* pnow;
#if !defined(_WIN32)
    struct tm now;
    pnow = localtime_r(&time_value, &now);
#else
    pnow = localtime(&time_value);  // NOLINT(*)
#endif
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d", pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
#endif
    return buffer_;
#else
    return "";
#endif  // _LIBCPP_SGX_CONFIG
  }

 private:
  char buffer_[9];
};

// class
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {
  }
  NullStream(const NullStream&) = delete;
};

template <class T>
MATXSCRIPT_ALWAYS_INLINE constexpr NullStream& operator<<(NullStream& os, const T&) {
  return os;
}

/*
 * The LoggingLevel is the same as Python builtin logging level
 */
namespace LoggingLevel {
static constexpr int64_t FATAL = 50;
static constexpr int64_t ERROR = 40;
static constexpr int64_t WARNING = 30;
static constexpr int64_t WARN = WARNING;
static constexpr int64_t INFO = 20;
static constexpr int64_t DEBUG = 10;
static constexpr int64_t NOTSET = 0;
};  // namespace LoggingLevel

extern NullStream null_stream;

MATX_DLL void SetLoggingLevel(int64_t level);

MATX_DLL int64_t GetLoggingLevel();

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
class LogMessage {
 public:
  LogMessage(const char* file, int line) : LogMessage(file, line, LoggingLevel::INFO) {
  }

  LogMessage(const char* file, int line, int64_t level)
      : log_stream_((GetLoggingLevel() > level) ? null_stream : std::cout) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":" << line << ": ";
  }
  ~LogMessage() {
    log_stream_ << '\n';
  }
  std::ostream& stream() {
    return log_stream_;
  }

 protected:
  std::ostream& log_stream_;

 private:
  DateLogger pretty_date_;
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};
#else
class DummyOStream {
 public:
  template <typename T>
  DummyOStream& operator<<(T _) {
    return *this;
  }
  inline std::string str() {
    return "";
  }
};
class LogMessage {
 public:
  LogMessage(const char* file, int line) : log_stream_() {
  }
  LogMessage(const char* file, int line, int64_t level) : log_stream_() {
  }
  DummyOStream& stream() {
    return log_stream_;
  }

 protected:
  DummyOStream log_stream_;

 private:
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};
#endif  // _LIBCPP_SGX_NO_IOSTREAMS

#if defined(_LIBCPP_SGX_NO_IOSTREAMS)
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) : LogMessage(file, line) {
  }
  ~LogMessageFatal() {
    abort();
  }

 private:
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#elif MATXSCRIPT_LOG_FATAL_THROW == 0
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) : LogMessage(file, line) {
  }
  ~LogMessageFatal() {
    if (ENV_ENABLE_MATX_LOG_STACK_TRACE) {
      log_stream_ << "\n" << StackTrace(1, LogStackTraceLevel()) << "\n";
    }
    abort();
  }

 private:
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#else
class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    Entry::ThreadLocal()->Init(file, line);
  }
  std::ostringstream& stream() {
    return Entry::ThreadLocal()->log_stream;
  }
  MATXSCRIPT_NO_INLINE ~LogMessageFatal() MATXSCRIPT_THROW_EXCEPTION {
#if MATXSCRIPT_LOG_STACK_TRACE
    if (ENV_ENABLE_MATX_LOG_STACK_TRACE) {
      Entry::ThreadLocal()->log_stream << "\n" << StackTrace(1, LogStackTraceLevel()) << "\n";
    }
#endif
    throw Entry::ThreadLocal()->Finalize();
  }

 private:
  struct Entry {
    std::ostringstream log_stream;
    MATXSCRIPT_NO_INLINE void Init(const char* file, int line) {
      DateLogger date;
      log_stream.str("");
      log_stream.clear();
      log_stream << "[" << date.HumanDate() << "] " << file << ":" << line << ": ";
    }
    ::matxscript::runtime::Error Finalize() {
#if MATXSCRIPT_LOG_BEFORE_THROW
      LOG(ERROR) << log_stream.str();
#endif
      return ::matxscript::runtime::Error(log_stream.str());
    }
    MATXSCRIPT_NO_INLINE static Entry* ThreadLocal() {
      static thread_local Entry* result = new Entry();
      return result;
    }
  };
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {
  }
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
#if !defined(_LIBCPP_SGX_NO_IOSTREAMS)
  void operator&(std::ostream&) {
  }
#endif
};

}  // namespace runtime
}  // namespace matxscript
