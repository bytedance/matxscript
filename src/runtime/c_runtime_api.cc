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

/*!
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include <matxscript/runtime/c_runtime_api.h>

#include "matxscript/runtime/container/string_helper.h"
#include "runtime_base.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/c_backend_api.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/function.h>
#include <matxscript/runtime/generic/generic_funcs.h>
#include <matxscript/runtime/object_internal.h>
#include <matxscript/runtime/regex/regex_ref.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/thread_local.h>
#include "matxscript/ir/_base/string_ref.h"

namespace matxscript {
namespace runtime {

//--------------------------------------------------------
// Error handling mechanism
// -------------------------------------------------------
// Standard error message format, {} means optional
//--------------------------------------------------------
// {error_type:} {message0}
// {message1}
// {message2}
// {Stack trace:}    // stack traces follow by this line
//   {trace 0}       // two spaces in the begining.
//   {trace 1}
//   {trace 2}
//--------------------------------------------------------
/*!
 * \brief Normalize error message
 *
 *  Parse them header generated by by LOG(FATAL) and CHECK
 *  and reformat the message into the standard format.
 *
 *  This function will also merge all the stack traces into
 *  one trace and trim them.
 *
 * \param err_msg The error message.
 * \return normalized message.
 */
std::string NormalizeError(std::string err_msg) {
  // ------------------------------------------------------------------------
  // log with header, {} indicates optional
  //-------------------------------------------------------------------------
  // [timestamp] file_name:line_number: {check_msg:} {error_type:} {message0}
  // {message1}
  // Stack trace:
  //   {stack trace 0}
  //   {stack trace 1}
  //-------------------------------------------------------------------------
  // Normalzied version
  //-------------------------------------------------------------------------
  // error_type: check_msg message0
  // {message1}
  // Stack trace:
  //   File file_name, line lineno
  //   {stack trace 0}
  //   {stack trace 1}
  //-------------------------------------------------------------------------
  int line_number = 0;
  std::istringstream is(err_msg);
  std::string line, file_name, error_type, check_msg;

  // Parse log header and set the fields,
  // Return true if it the log is in correct format,
  // return false if something is wrong.
  auto parse_log_header = [&]() {
    // skip timestamp
    if (is.peek() != '[') {
      getline(is, line);
      return true;
    }
    if (!(is >> line))
      return false;
    // get filename
    while (is.peek() == ' ')
      is.get();
#ifdef _MSC_VER  // handle volume separator ":" in Windows path
    std::string drive;
    if (!getline(is, drive, ':'))
      return false;
    if (!getline(is, file_name, ':'))
      return false;
    file_name = drive + ":" + file_name;
#else
    if (!getline(is, file_name, ':'))
      return false;
#endif
    // get line number
    if (!(is >> line_number))
      return false;
    // get rest of the message.
    while (is.peek() == ' ' || is.peek() == ':')
      is.get();
    if (!getline(is, line))
      return false;
    // detect check message, rewrite to remote extra :
    if (line.compare(0, 13, "Check failed:") == 0) {
      size_t end_pos = line.find(':', 13);
      if (end_pos == std::string::npos)
        return false;
      check_msg = line.substr(0, end_pos + 1) + ' ';
      line = line.substr(end_pos + 1);
    }
    return true;
  };
  // if not in correct format, do not do any rewrite.
  if (!parse_log_header())
    return err_msg;
  // Parse error type.
  {
    size_t start_pos = 0, end_pos;
    for (; start_pos < line.length() && line[start_pos] == ' '; ++start_pos) {
    }
    for (end_pos = start_pos; end_pos < line.length(); ++end_pos) {
      char ch = line[end_pos];
      if (ch == ':') {
        error_type = line.substr(start_pos, end_pos - start_pos);
        break;
      }
      // [A-Z0-9a-z_.]
      if (!std::isalpha(ch) && !std::isdigit(ch) && ch != '_' && ch != '.')
        break;
    }
    if (error_type.length() != 0) {
      // if we successfully detected error_type: trim the following space.
      for (start_pos = end_pos + 1; start_pos < line.length() && line[start_pos] == ' ';
           ++start_pos) {
      }
      line = line.substr(start_pos);
    } else {
      // did not detect error_type, use default value.
      line = line.substr(start_pos);
      error_type = "MATXError";
    }
  }
  // Seperate out stack trace.
  auto py_info_pat = regex::RegexPattern::Load(R"(^File ".*\.py", line \d+)");
  std::ostringstream os;
  os << error_type << ": ";
  if (!py_info_pat->Find(line)) {
    os << "File \"" << file_name << "\", line " << line_number << ", in\n";
  }
  os << check_msg << line << '\n';

  bool trace_mode = false;
  std::vector<std::string> stack_trace;
  while (getline(is, line)) {
    if (trace_mode) {
      if (line.compare(0, 2, "  ") == 0) {
        stack_trace.push_back(line);
      } else {
        trace_mode = false;
        // remove EOL trailing stacktrace.
        if (line.length() == 0)
          continue;
      }
    }
    if (!trace_mode) {
      if (line.compare(0, 11, "Stack trace") == 0) {
        trace_mode = true;
      } else {
        os << line << '\n';
      }
    }
  }
  if (stack_trace.size() != 0 || file_name.length() != 0) {
    os << "Stack trace:\n";
    // Print out stack traces, optionally trim the c++ traces
    // about the frontends (as they will be provided by the frontends).
    bool ffi_boundary = false;
    for (const auto& line : stack_trace) {
      // Heuristic to detect python ffi.
      if (line.find("libffi.so") != std::string::npos ||
          line.find("core.cpython") != std::string::npos) {
        ffi_boundary = true;
      }
      // If the backtrace is not c++ backtrace with the prefix "  [bt]",
      // then we can stop trimming.
      if (ffi_boundary && line.compare(0, 6, "  [bt]") != 0) {
        ffi_boundary = false;
      }
      if (!ffi_boundary) {
        os << line << '\n';
      }
      // The line after MATXFuncCall cound be in FFI.
      if (line.find("(MATXFuncCall") != std::string::npos) {
        ffi_boundary = true;
      }
    }
  }
  return os.str();
}

}  // namespace runtime
}  // namespace matxscript

using namespace ::matxscript::runtime;

struct MATXRuntimeEntry {
  std::string last_error;
};

typedef ::matxscript::runtime::ThreadLocalStore<MATXRuntimeEntry> MATXAPIRuntimeStore;

const char* MATXScriptAPIGetLastError() {
  return MATXAPIRuntimeStore::Get()->last_error.c_str();
}

int MATXScriptAPIHandleException(const std::runtime_error& e) {
  MATXScriptAPISetLastError(NormalizeError(e.what()).c_str());
  return -1;
}

void MATXScriptAPISetLastError(const char* msg) {
  MATXAPIRuntimeStore::Get()->last_error = msg;
}

int MATXScriptModLoadFromFile(const char* file_name,
                              const char* format,
                              MATXScriptModuleHandle* out) {
  API_BEGIN();
  RTValue ret;
  ret = Module::LoadFromFile(file_name, format);
  MATXScriptAny val;
  ret.MoveToCHost(&val);
  *out = val.data.v_handle;
  API_END();
}

int MATXScriptModImport(MATXScriptModuleHandle mod, MATXScriptModuleHandle dep) {
  API_BEGIN();
  ObjectInternal::GetModuleNode(mod)->Import(GetRef<Module>(ObjectInternal::GetModuleNode(dep)));
  API_END();
}

int MATXScriptModGetFunction(MATXScriptModuleHandle mod,
                             const char* func_name,
                             int query_imports,
                             MATXScriptFunctionHandle* out) {
  API_BEGIN();
  auto pf = ObjectInternal::GetModuleNode(mod)->GetFunction(func_name, query_imports != 0);
  if (pf != nullptr) {
    *out = new NativeFunction(pf);
  } else {
    *out = nullptr;
  }
  API_END();
}

int MATXScriptModFree(MATXScriptModuleHandle mod) {
  return MATXScriptObjectFree(mod);
}

int MATXScriptFuncFree(MATXScriptFunctionHandle func) {
  API_BEGIN();
  delete static_cast<NativeFunction*>(func);
  API_END();
}

int MATXScriptFuncCall_PYTHON_C_API(MATXScriptFunctionHandle func,
                                    MATXScriptAny* arg_values,
                                    int num_args,
                                    MATXScriptAny* ret_val) {
  API_BEGIN();

  std::vector<RTView> args;
  args.reserve(num_args);
  for (int i = 0; i < num_args; ++i) {
    args.push_back(RTView(arg_values[i]));
  }
  RTValue rv = (*static_cast<const NativeFunction*>(func))(PyArgs(args.data(), args.size()));
  // handle return string.
  switch (rv.type_code()) {
    case TypeIndex::kRuntimeDataType: {
      auto ds = DLDataType2String(rv.As<DataType>());
      String(ds.data(), ds.size()).decode().MoveTo(ret_val);
    } break;
#ifdef MATX_RUNTIME_ENABLE_STRINGREF
    case TypeIndex::kRuntimeStringRef: {
      auto ref = rv.AsObjectRefNoCheck<StringRef>();
      String(ref.data(), ref.size()).MoveTo(ret_val);
    } break;
#endif
    default: {
      rv.MoveToCHost(ret_val);
    } break;
  }
  API_END();
}

int MATXScriptAPIDLDataTypeToString(DLDataType dtype, char* buffer, int* size) {
  API_BEGIN();
  auto s = DLDataType2String(dtype);
  MXCHECK(*size > s.size()) << "DLDataType buffer overflow";
  memcpy(buffer, s.data(), s.size() + 1);
  *size = s.size();
  API_END();
}

int MATXScriptRuntimeRetain(MATXScriptAny* value) {
  API_BEGIN();
  MATXScriptAny dest;
  RTValue::CopyFromCHostToCHost(value, &dest);
  *value = dest;
  API_END();
}

int MATXScriptRuntimeDestroyN(MATXScriptAny* values, int num) {
  API_BEGIN();
  for (int i = 0; i < num; ++i) {
    RTValue::DestroyCHost(values + i);
  }
  API_END();
}

int MATXScriptRuntimeDestroy(MATXScriptAny* value) {
  API_BEGIN();
  RTValue::DestroyCHost(value);
  API_END();
}

int MATXScriptPipelineTXSessionRun(void* session_handle,
                                   const char** keys,
                                   MATXScriptAny* arg_values,
                                   int num_args,
                                   int move_mode,
                                   int* num_rets,
                                   MATXScriptAny* ret_val) {
  API_BEGIN();
  auto* sess = static_cast<TXSession*>(session_handle);
  std::unordered_map<std::string, RTValue> feed_dict;
  feed_dict.reserve(num_args);
  if (move_mode) {
    for (int i = 1; i < num_args; ++i) {
      feed_dict.emplace(std::string(keys[i]), RTValue::MoveFromCHost(arg_values + i));
    }
  } else {
    for (int i = 1; i < num_args; ++i) {
      feed_dict.emplace(std::string(keys[i]), RTValue::CopyFromCHost(arg_values + i));
    }
  }
  auto result = sess->Run(feed_dict);
  MXCHECK(result.size() <= *num_rets) << "[MATXScriptPipelineTXSessionRun] ret_val cache overflow";
  *num_rets = result.size();
  for (auto i = 0; i < result.size(); ++i) {
    result[i].second.MoveToCHost(ret_val + i);
  }
  API_END();
}

int MATXScriptPipelineOpKernelCall(void* op_handle,
                                   MATXScriptAny* arg_values,
                                   int num_args,
                                   int move_mode,
                                   MATXScriptAny* ret_val) {
  API_BEGIN();
  std::vector<RTValue> op_args;
  op_args.reserve(num_args);
  if (move_mode) {
    for (int i = 0; i < num_args; ++i) {
      op_args.push_back(RTValue::MoveFromCHost(arg_values + i));
    }
  } else {
    for (int i = 0; i < num_args; ++i) {
      op_args.push_back(RTValue::CopyFromCHost(arg_values + i));
    }
  }
  auto* op = static_cast<OpKernel*>(op_handle);
  RTValue result = op->Process(PyArgs(op_args.data(), op_args.size()));
  result.MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeMakeString(const char* buffer, size_t size, MATXScriptAny* ret_val) {
  API_BEGIN();
  RTValue(String(buffer, size)).MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeMakeUnicode(const char* buffer, size_t size, MATXScriptAny* ret_val) {
  API_BEGIN();
  RTValue(StringHelper::Decode({buffer, size})).MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeUnicodeEncode(MATXScriptAny* arg_value, MATXScriptAny* ret_val) {
  API_BEGIN();
  RTValue(UnicodeHelper::Encode(UnicodeHelper::AsView(arg_value))).MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeMakeList(MATXScriptAny* arg_values,
                              int num_args,
                              int move_mode,
                              MATXScriptAny* ret_val) {
  API_BEGIN();
  List result;
  result.reserve(num_args);
  if (move_mode) {
    for (int i = 0; i < num_args; ++i) {
      result.append(RTValue::MoveFromCHost(arg_values + i));
    }
  } else {
    for (int i = 0; i < num_args; ++i) {
      result.append(RTValue::CopyFromCHost(arg_values + i));
    }
  }
  RTValue(std::move(result)).MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeGetListSize(MATXScriptAny* arg_value, int64_t* size) {
  API_BEGIN();
  if (auto d = static_cast<ListNode*>(arg_value->data.v_handle)) {
    *size = d->size();
  } else {
    *size = 0;
  }
  API_END();
}

int MATXScriptRuntimeGetListItems(MATXScriptAny* arg_value,
                                  int move_mode,
                                  int64_t* num_rets,
                                  MATXScriptAny* ret_val) {
  API_BEGIN();
  if (move_mode) {
    auto container = RTValue::MoveFromCHost(arg_value).MoveToObjectRef<List>();
    *num_rets = container.size();
    auto i = 0;
    for (auto& item : container) {
      item.CopyToCHost(&ret_val[i++]);
    }
  } else {
    auto container = RTValue::CopyFromCHost(arg_value).MoveToObjectRef<List>();
    *num_rets = container.size();
    auto i = 0;
    for (auto& item : container) {
      item.CopyToCHost(&ret_val[i++]);
    }
  }
  API_END();
}

int MATXScriptRuntimeMakeDict(MATXScriptAny* arg_values,
                              int num_args,
                              int move_mode,
                              MATXScriptAny* ret_val) {
  API_BEGIN();
  Dict result;
  result.reserve((num_args + 1) / 2);
  if (move_mode) {
    for (int i = 0; i < num_args; i += 2) {
      result.emplace(RTValue::MoveFromCHost(arg_values + i),
                     RTValue::MoveFromCHost(arg_values + i + 1));
    }
  } else {
    for (int i = 0; i < num_args; ++i) {
      result.emplace(RTValue::CopyFromCHost(arg_values + i),
                     RTValue::CopyFromCHost(arg_values + i + 1));
    }
  }
  RTValue(std::move(result)).MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeGetDictSize(MATXScriptAny* arg_value, int64_t* size) {
  API_BEGIN();
  if (auto d = static_cast<DictNode*>(arg_value->data.v_handle)) {
    *size = d->size();
  } else {
    *size = 0;
  }
  API_END();
}

int MATXScriptRuntimeGetDictItems(MATXScriptAny* arg_value,
                                  int move_mode,
                                  int64_t* num_rets,
                                  MATXScriptAny* ret_val) {
  API_BEGIN();
  if (move_mode) {
    auto container = RTValue::MoveFromCHost(arg_value).MoveToObjectRef<Dict>();
    *num_rets = container.size() * 2;
    auto i = 0;
    for (auto& item : container.items()) {
      item.first.CopyToCHost(&ret_val[i++]);
      item.second.CopyToCHost(&ret_val[i++]);
    }
  } else {
    auto container = RTValue::CopyFromCHost(arg_value).MoveToObjectRef<Dict>();
    *num_rets = container.size() * 2;
    auto i = 0;
    for (auto& item : container.items()) {
      item.first.CopyToCHost(&ret_val[i++]);
      item.second.CopyToCHost(&ret_val[i++]);
    }
  }
  API_END();
}

int MATXScriptRuntimeMakeSet(MATXScriptAny* arg_values,
                             int num_args,
                             int move_mode,
                             MATXScriptAny* ret_val) {
  API_BEGIN();
  Set result;
  result.reserve(num_args);
  if (move_mode) {
    for (int i = 0; i < num_args; ++i) {
      result.add(RTValue::MoveFromCHost(arg_values + i));
    }
  } else {
    for (int i = 0; i < num_args; ++i) {
      result.add(RTValue::CopyFromCHost(arg_values + i));
    }
  }
  RTValue(std::move(result)).MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeGetSetSize(MATXScriptAny* arg_value, int64_t* size) {
  API_BEGIN();
  if (auto d = static_cast<SetNode*>(arg_value->data.v_handle)) {
    *size = d->size();
  } else {
    *size = 0;
  }
  API_END();
}

int MATXScriptRuntimeGetSetItems(MATXScriptAny* arg_value,
                                 int move_mode,
                                 int64_t* num_rets,
                                 MATXScriptAny* ret_val) {
  API_BEGIN();
  if (move_mode) {
    auto container = RTValue::MoveFromCHost(arg_value).MoveToObjectRef<Set>();
    *num_rets = container.size();
    auto i = 0;
    for (auto& item : container) {
      item.CopyToCHost(&ret_val[i++]);
    }
  } else {
    auto container = RTValue::CopyFromCHost(arg_value).MoveToObjectRef<Set>();
    *num_rets = container.size();
    auto i = 0;
    for (auto& item : container) {
      item.CopyToCHost(&ret_val[i++]);
    }
  }
  API_END();
}

int MATXScriptRuntimeMakeTuple(MATXScriptAny* arg_values,
                               int num_args,
                               int move_mode,
                               MATXScriptAny* ret_val) {
  API_BEGIN();
  std::vector<RTValue> result;
  result.reserve(num_args);
  if (move_mode) {
    for (int i = 0; i < num_args; ++i) {
      result.push_back(RTValue::MoveFromCHost(arg_values + i));
    }
  } else {
    for (int i = 0; i < num_args; ++i) {
      result.push_back(RTValue::CopyFromCHost(arg_values + i));
    }
  }
  RTValue(Tuple(std::make_move_iterator(result.begin()), std::make_move_iterator(result.end())))
      .MoveToCHost(ret_val);
  API_END();
}

int MATXScriptRuntimeGetTupleSize(MATXScriptAny* arg_value, int64_t* size) {
  API_BEGIN();
  if (auto d = static_cast<TupleNode*>(arg_value->data.v_handle)) {
    *size = d->size;
  } else {
    *size = 0;
  }
  API_END();
}

int MATXScriptRuntimeGetTupleItems(MATXScriptAny* arg_value,
                                   int move_mode,
                                   int64_t* num_rets,
                                   MATXScriptAny* ret_val) {
  API_BEGIN();
  if (move_mode) {
    auto container = RTValue::MoveFromCHost(arg_value).MoveToObjectRef<Tuple>();
    *num_rets = container.size();
    for (auto i = 0; i < *num_rets; ++i) {
      container[i].CopyToCHost(&ret_val[i]);
    }
  } else {
    auto container = RTValue::CopyFromCHost(arg_value).MoveToObjectRef<Tuple>();
    *num_rets = container.size();
    for (auto i = 0; i < *num_rets; ++i) {
      container[i].CopyToCHost(&ret_val[i]);
    }
  }
  API_END();
}

int MATXScriptCFuncSetReturn(MATXScriptValueHandle ret, MATXScriptAny* value, int num_ret) {
  API_BEGIN();
  MXCHECK_EQ(num_ret, 1);
  RTValue* rv = static_cast<RTValue*>(ret);
  *rv = RTView(value[0]);
  API_END();
}

int MATXScriptFuncCreateFromCFunc(MATXScriptPackedCFunc func,
                                  void* resource_handle,
                                  MATXScriptPackedCFuncFinalizer fin,
                                  MATXScriptFunctionHandle* out) {
  API_BEGIN();
  if (fin == nullptr) {
    *out = new NativeFunction([func, resource_handle](PyArgs args) -> RTValue {
      std::vector<MATXScriptAny> c_args;
      c_args.reserve(args.size());
      for (auto& val : args) {
        c_args.push_back(val.value());
      }
      RTValue rv;
      int ret = func(c_args.data(), args.size(), &rv, resource_handle);
      if (ret != 0) {
        throw ::matxscript::runtime::Error(MATXScriptAPIGetLastError() + std::string("\n") +
                                           ::matxscript::runtime::StackTrace());
      }
      return rv;
    });
  } else {
    // wrap it in a shared_ptr, with fin as deleter.
    // so fin will be called when the lambda went out of scope.
    std::shared_ptr<void> rpack(resource_handle, fin);
    *out = new NativeFunction([func, rpack](PyArgs args) -> RTValue {
      std::vector<MATXScriptAny> c_args;
      c_args.reserve(args.size());
      for (auto& val : args) {
        c_args.push_back(val.value());
      }
      RTValue rv;
      int ret = func(c_args.data(), args.size(), &rv, rpack.get());
      if (ret != 0) {
        throw ::matxscript::runtime::Error(MATXScriptAPIGetLastError() + std::string("\n") +
                                           ::matxscript::runtime::StackTrace());
      }
      return rv;
    });
  }
  API_END();
}

/******************************************************************************
 * Function Register
 *****************************************************************************/

/*! \brief entry to to easily hold returning information */
struct MATXFuncThreadLocalEntry {
  /*! \brief result holder for returning string pointers */
  std::vector<const char*> ret_vec_charp;
};

/*! \brief Thread local store that can be used to hold return values. */
typedef ::matxscript::runtime::ThreadLocalStore<MATXFuncThreadLocalEntry> MATXFuncThreadLocalStore;

int MATXScriptFuncRegisterGlobal(const char* name, MATXScriptFunctionHandle f, int override) {
  API_BEGIN();
  ::matxscript::runtime::FunctionRegistry::Register(name, override != 0)
      .set_body(*static_cast<::matxscript::runtime::NativeFunction*>(f));
  API_END();
}

int MATXScriptFuncGetGlobal(const char* name, MATXScriptFunctionHandle* out) {
  API_BEGIN();
  const ::matxscript::runtime::NativeFunction* fp =
      ::matxscript::runtime::FunctionRegistry::Get(name);
  if (fp != nullptr) {
    *out = new ::matxscript::runtime::NativeFunction(*fp);  // NOLINT(*)
  } else {
    *out = nullptr;
  }
  API_END();
}

int MATXScriptFuncListGlobalNames(int* out_size, const char*** out_array) {
  API_BEGIN();
  MATXFuncThreadLocalEntry* ret = MATXFuncThreadLocalStore::Get();
  auto ret_vec_str = ::matxscript::runtime::FunctionRegistry::ListNames();
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret_vec_str[i].data());
  }
  *out_array = ::matxscript::runtime::BeginPtr(ret->ret_vec_charp);
  *out_size = static_cast<int>(ret_vec_str.size());
  API_END();
}

int MATXScriptStreamCreate(int device_type, int device_id, MATXScriptStreamHandle* out) {
  API_BEGIN();
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = DeviceAPI::Get(ctx)->CreateStream(ctx);
  API_END();
}

int MATXScriptStreamFree(int device_type, int device_id, MATXScriptStreamHandle stream) {
  API_BEGIN();
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPI::Get(ctx)->FreeStream(ctx, stream);
  API_END();
}

int MATXScriptSetCurrentThreadStream(int device_type,
                                     int device_id,
                                     MATXScriptStreamHandle handle) {
  API_BEGIN();
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPI::Get(ctx)->SetCurrentThreadStream(ctx, std::shared_ptr<void>(handle, [](void*) {}));
  API_END();
}

int MATXScriptSynchronize(int device_type, int device_id, MATXScriptStreamHandle stream) {
  API_BEGIN();
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPI::Get(ctx)->StreamSync(ctx, stream);
  API_END();
}

int MATXScriptStreamStreamSynchronize(int device_type,
                                      int device_id,
                                      MATXScriptStreamHandle src,
                                      MATXScriptStreamHandle dst) {
  API_BEGIN();
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPI::Get(ctx)->SyncStreamFromTo(ctx, src, dst);
  API_END();
}

int MATXScriptDeviceAllocDataSpace(
    DLContext ctx, size_t nbytes, size_t alignment, DLDataType type_hint, void** out_data) {
  API_BEGIN();
  out_data[0] = DeviceAPI::Get(ctx)->Alloc(ctx, nbytes, alignment, type_hint);
  API_END();
}

int MATXScriptDeviceFreeDataSpace(DLContext ctx, void* ptr) {
  API_BEGIN();
  DeviceAPI::Get(ctx)->Free(ctx, ptr);
  API_END();
}

int MATXScriptNDArrayToDLPack(MATXScriptAny* value, DLManagedTensor** dlpack) {
  API_BEGIN();
  auto ndarray = RTValue::MoveFromCHost(value).MoveToObjectRef<NDArray>();
  DLManagedTensor* ret = ndarray.ToDLPack();
  *dlpack = ret;
  API_END();
}

int MATXScriptNDArrayFromDLPack(void* dlpack, MATXScriptAny* value) {
  API_BEGIN();
  DLManagedTensor* dlm_tensor = static_cast<DLManagedTensor*>(dlpack);
  RTValue(NDArray::FromDLPack(dlm_tensor)).MoveToCHost(value);
  API_END();
}

int MATXScriptSetDeviceDriverError(int device_type, const char* msg) {
  API_BEGIN();
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = 0;
  DeviceAPI::SetErrorMessage(ctx, String(msg));
  API_END();
}
