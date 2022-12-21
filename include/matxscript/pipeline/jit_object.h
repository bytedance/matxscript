// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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
#ifndef MATXSCRIPT_RUNTIME_JIT_OBJECT_H
#define MATXSCRIPT_RUNTIME_JIT_OBJECT_H

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/function.h>
#include <matxscript/runtime/module.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class TXSession;
class JitObject;
using JitObjectPtr = std::shared_ptr<JitObject>;

class JitObject : public OpKernel {
  using NativeMethod = std::function<RTValue(void* self, PyArgs args)>;

 public:
  // simple function schema for call check
  struct FuncParam {
    // size_t type;
    String name;
    int32_t type_code;
    FuncParam() : name(), type_code() {
    }
    FuncParam(String name, int32_t type_code) : name(std::move(name)), type_code(type_code) {
    }
    ~FuncParam() = default;
    static FuncParam FromDict(const Dict& config);
    Dict ToDict() const;
  };

  struct FuncMeta {
    String name;
    bool bound_self;
    std::vector<FuncParam> args;
    List defaults;
    FuncMeta() : name(), bound_self(false), args() {
    }
    FuncMeta(std::string name, bool bound_self, std::vector<FuncParam> args, List defaults)
        : name(std::move(name)),
          bound_self(bound_self),
          args(std::move(args)),
          defaults(std::move(defaults)) {
    }
    ~FuncMeta() = default;
    static FuncMeta FromDict(const Dict& config);
    Dict ToDict() const;
  };

  struct ClassMeta {
    String name;
    int32_t len_slots;
    FuncMeta init_func;
    std::vector<RTValue> init_args;
    std::vector<FuncMeta> member_funcs;
    ClassMeta() : name(), len_slots(-1), init_func(), init_args(), member_funcs() {
    }
    ClassMeta(String name,
              int32_t len_slots,
              FuncMeta init_func,
              std::vector<RTValue> init_args,
              std::vector<FuncMeta> member_funcs)
        : name(std::move(name)),
          len_slots(len_slots),
          init_func(std::move(init_func)),
          init_args(std::move(init_args)),
          member_funcs(std::move(member_funcs)) {
    }
    ~ClassMeta() = default;
    static ClassMeta FromDict(const Dict& config);
    Dict ToDict() const;
  };

  // resource options
  struct Options {
    String dso_path;
    String dso_path_cxx11;
    ClassMeta class_info;
    FuncMeta func_info;
    // need bundle save init_args name when it is a file location
    std::vector<String> need_bundle;
    std::vector<std::pair<String, String>> captures;
    bool is_class = false;
    bool share = true;
    int64_t py_source_line_ = -1;
    String py_source_file_;

    static Options FromDict(const Dict& config);
    Dict ToDict() const;
  };

 public:
  // constructor
  ~JitObject() override = default;
  JitObject() = default;

  const UserDataRef& self() const {
    return self_;
  }
  std::pair<NativeFunction, const FuncMeta*> GetFunction(string_view name);

  const String& PyObjectName() const;

 public:
  // override op kernel method
  void Init() override;

  int Bundle(string_view folder) override;

  const ska::flat_hash_map<string_view, NativeMethod>* GetFunctionTable() const {
    return &function_table_;
  }

  RTValue generic_call_attr(string_view func_name, PyArgs args);

 private:
  static NativeMethod MakeNativeFunc(const FuncMeta& meta,
                                     UserDataRef self,
                                     MATXScriptBackendPackedCFunc c_func);

 private:
  Options options_;
  // module
  Module module_;
  UserDataRef self_;
  ska::flat_hash_map<string_view, NativeMethod> function_table_;
  friend class JitOp;
  friend class TXSession;
};

JitObjectPtr check_get_jit_object(const UserDataRef& ud);
JitObjectPtr try_get_jit_object(const UserDataRef& ud);

}  // namespace runtime
}  // namespace matxscript

#endif  // MATXSCRIPT_RUNTIME_JIT_OBJECT_H
