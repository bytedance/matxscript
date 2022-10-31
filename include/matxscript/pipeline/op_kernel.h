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
#ifndef MATXSCRIPT_RUNTIME_PIPELINE_OP_KERNEL_H
#define MATXSCRIPT_RUNTIME_PIPELINE_OP_KERNEL_H

#include <functional>
#include <tuple>
#include <vector>

#include <matxscript/pipeline/attributes.h>
#include <matxscript/pipeline/global_unique_index.h>
#include <matxscript/runtime/native_object_registry.h>

namespace matxscript {
namespace runtime {

class TXSession;
class OpKernel;
using OpKernelPtr = std::shared_ptr<OpKernel>;

class OpKernel {
 public:
  OpKernel() = default;
  virtual ~OpKernel() = default;
  OpKernel(const OpKernel&) = default;
  OpKernel(OpKernel&&) = default;
  OpKernel& operator=(const OpKernel&) = default;
  OpKernel& operator=(OpKernel&&) = default;

  template <class T>
  static std::unique_ptr<T> Create(const Dict& config) {
    Attributes attrs = Attributes::FromDict(config);
    auto op = std::make_unique<T>();
    op->Initialize(std::move(attrs));
    return std::move(attrs);
  }

 public:
  MATXSCRIPT_ALWAYS_INLINE void CheckArgs(size_t arguments_size, size_t expect_size) const {
    MXCHECK_EQ(arguments_size, expect_size) << "[" << class_name_ << "] Expect " << expect_size
                                            << " arguments but get " << arguments_size;
  }

  virtual void Init() {
  }
  virtual RTValue Process(PyArgs inputs) const {
    MXCHECK(false) << "[" << class_name_
                   << "] NotImplementedError: The Process method is not implemented";
    return None;
  }
  virtual int Bundle(string_view folder) {
    return 0;
  }

  inline bool HasAttr(string_view key) {
    return attributes_.HasAttr(key);
  }
  template <class U>
  inline U GetAttr(string_view key, const U& default_val = U{}) const {
    return attributes_.GetAttr(key, default_val);
  }
  template <class U>
  inline void SetAttr(string_view key, U&& val) {
    return attributes_.SetAttr(key, std::forward<U>(val));
  }

  String BundlePath(string_view location, string_view folder) const;

 public:
  void Initialize(Attributes attrs);

  string_view ClassName() const {
    return class_name_;
  }
  const String& GetName() const {
    return name_;
  }

  void SetBelongTo(TXSession* belong_to);

  OpKernelPtr GetOpImpl(string_view cls, string_view name);

 protected:
  int device_ = NONE_DEVICE;
  string_view class_name_;
  String name_;
  String resource_path_;
  Attributes attributes_;
  std::vector<OpKernelPtr> sub_ops_;
  TXSession* belong_to_ = nullptr;
  friend class SymbolicExecutor;
  friend class TXSession;
  friend UserDataRef make_op_kernel(string_view, PyArgs args, TXSession* sess);
  friend struct UserDataMutator;
};

OpKernelPtr check_get_op_kernel(const UserDataRef& ud);
OpKernelPtr try_get_op_kernel(const UserDataRef& ud);
UserDataRef make_userdata(OpKernelPtr op_ptr);
UserDataRef make_op_kernel(string_view class_name, PyArgs args, TXSession* sess = nullptr);

inline RTValue op_kernel_call(void* self, PyArgs args) {
  OpKernel* op = reinterpret_cast<OpKernel*>(self);
  return op->Process(args);
}

}  // namespace runtime
}  // namespace matxscript

#define MATX_REGISTER_NATIVE_OP(ClassName)                                              \
  MATX_REGISTER_NATIVE_OBJECT(ClassName)                                                \
      .SetConstructor([](::matxscript::runtime::PyArgs args) -> std::shared_ptr<void> { \
        MXCHECK(args.size() == 1 && args[0].IsObjectRef<Dict>())                        \
            << "[NativeOp:" << #ClassName                                               \
            << "] only need one dict type arg, but get arg num: " << args.size()        \
            << ", args[0] type: " << args[0].type_name();                               \
        auto op = std::static_pointer_cast<OpKernel>(std::make_shared<ClassName>());    \
        return std::move(op);                                                           \
      })                                                                                \
      .RegisterFunction("__call__", op_kernel_call)

#endif  // MATXSCRIPT_RUNTIME_PIPELINE_OP_KERNEL_H
