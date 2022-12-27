// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
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
#include <matxscript/runtime/demangle.h>
#include <matxscript/runtime/logging.h>

#include <mutex>

namespace matxscript {
namespace runtime {

/*
 * The default logging_level is set to WARNING
 */

static int64_t logging_level = LoggingLevel::WARNING;

NullStream null_stream;

MATX_DLL void SetLoggingLevel(int64_t level) {
  logging_level = level;
}

MATX_DLL int64_t GetLoggingLevel() {
  return logging_level;
}

static int GET_ENV_MATXSCRIPT_LOG_STACK_TRACE() {
  if (auto var = std::getenv("MATXSCRIPT_LOG_STACK_TRACE")) {
    return std::atoi(var);
  }
  return 1;
}

bool ENV_ENABLE_MATX_LOG_STACK_TRACE = GET_ENV_MATXSCRIPT_LOG_STACK_TRACE();

}  // namespace runtime
}  // namespace matxscript

#ifdef MATXSCRIPT_LOG_STACK_TRACE

#ifdef MATX_WITH_LIBBACKTRACE

#include <backtrace.h>
#include <cxxabi.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace matxscript {
namespace runtime {
namespace {

struct BacktraceInfo {
  std::vector<std::string> lines;
  size_t max_size;
  std::string error_message;
};

void BacktraceCreateErrorCallback(void* data, const char* msg, int errnum) {
}

backtrace_state* BacktraceCreate() {
  return backtrace_create_state(nullptr, 1, BacktraceCreateErrorCallback, nullptr);
}

static backtrace_state* _bt_state = BacktraceCreate();

std::string DemangleName(std::string name) {
  int status = 0;
  size_t length = name.size();
  std::unique_ptr<char, void (*)(void* __ptr)> demangled_name = {
      abi::__cxa_demangle(name.c_str(), nullptr, &length, &status), &std::free};
  if (demangled_name && status == 0 && length > 0) {
    return demangled_name.get();
  } else {
    return name;
  }
}

void BacktraceErrorCallback(void* data, const char* msg, int errnum) {
}

void BacktraceSyminfoCallback(
    void* data, uintptr_t pc, const char* symname, uintptr_t symval, uintptr_t symsize) {
  auto str = reinterpret_cast<std::string*>(data);
  if (symname != nullptr) {
    std::string tmp(symname, symsize);
    *str = DemangleName(tmp.c_str());
  } else {
    std::ostringstream s;
    s << "0x" << std::setfill('0') << std::setw(sizeof(uintptr_t) * 2) << std::hex << pc;
    *str = s.str();
  }
}

int BacktraceFullCallback(
    void* data, uintptr_t pc, const char* filename, int lineno, const char* symbol) {
  auto stack_trace = reinterpret_cast<BacktraceInfo*>(data);
  std::stringstream s;

  std::unique_ptr<std::string> symbol_str = std::make_unique<std::string>("<unknown>");
  if (symbol != nullptr) {
    *symbol_str = DemangleName(symbol);
  } else {
    backtrace_syminfo(
        _bt_state, pc, BacktraceSyminfoCallback, BacktraceErrorCallback, symbol_str.get());
  }
  s << *symbol_str;

  if (filename != nullptr) {
    s << std::endl << "        at " << filename;
    if (lineno != 0) {
      s << ":" << lineno;
    }
  }

  if (!(stack_trace->lines.size() == 0 &&
        (symbol_str->find("matxscript::runtime::StackTrace", 0) == 0 ||
         symbol_str->find("matxscript::runtime::LogMessageFatal", 0) == 0))) {
    stack_trace->lines.push_back(s.str());
  }
  std::vector<std::string> terminates = {
      "MATXPipelineTXSessionRun", "MATXPipelineOpKernelCall", "MATXFuncCall"};
  if (std::find(terminates.begin(), terminates.end(), *symbol_str) != terminates.end()) {
    return 1;
  }
  if (stack_trace->lines.size() >= stack_trace->max_size) {
    return 1;
  }
  return 0;
}
}  // namespace

std::string StackTrace(size_t start_frame, const size_t stack_size) {
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  BacktraceInfo bt;
  bt.max_size = stack_size;
  if (_bt_state == nullptr) {
    return "";
  }
  // libbacktrace eats memory if run on multiple threads at the same time, so we guard against it
  backtrace_full(_bt_state, 0, BacktraceFullCallback, BacktraceErrorCallback, &bt);

  std::ostringstream s;
  s << "Stack trace:\n";
  for (size_t i = 0; i < bt.lines.size(); i++) {
    s << "  " << i << ": " << bt.lines[i] << "\n";
  }
  return s.str();
}

}  // namespace runtime
}  // namespace matxscript

#else

#define MATX_EXECINFO_H <execinfo.h>
#include <cxxabi.h>
#include <sstream>
#include MATX_EXECINFO_H

namespace matxscript {
namespace runtime {

std::string StackTrace(size_t start_frame, const size_t stack_size) {
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  using std::string;
  std::ostringstream stacktrace_os;
  std::vector<void*> stack(stack_size);
  int nframes = backtrace(stack.data(), static_cast<int>(stack_size));
  if (start_frame < static_cast<size_t>(nframes)) {
    stacktrace_os << "Stack trace:\n";
  }
  char** msgs = backtrace_symbols(stack.data(), nframes);
  if (msgs != nullptr) {
    for (int frameno = start_frame; frameno < nframes; ++frameno) {
      string msg = ::matxscript::runtime::Demangle(msgs[frameno]);
      stacktrace_os << "  [bt] (" << frameno - start_frame << ") " << msg << "\n";
    }
  }
  free(msgs);
  string stack_trace = stacktrace_os.str();
  return stack_trace;
}

}  // namespace runtime
}  // namespace matxscript

#endif  // MATXSCRIPT_WITH_LIBBACKTRACE
#endif  // MATXSCRIPT_LOG_STACK_TRACE
