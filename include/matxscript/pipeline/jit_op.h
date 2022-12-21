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
#pragma once

#include <matxscript/pipeline/jit_object.h>
#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/module.h>

namespace matxscript {
namespace runtime {

// JitOp is a pipeline op with uniq function entry UserCodegenResource
class JitOp : public OpKernel {
 public:
  void Init() override;
  int Bundle(string_view folder) override;
  RTValue Process(PyArgs inputs) const override;

  RTValue generic_call_attr(string_view func_name, PyArgs args);

  String GetHumanName(bool with_debug_info) const;

 protected:
  String main_func_name_;
  String jit_object_name_;
  std::shared_ptr<JitObject> jit_object_;
  NativeFunction forward_func_;
  const JitObject::FuncMeta* func_meta_ = nullptr;
  UserDataRef self_;

  friend class NativeObject;
  friend class UserDataRef;
  friend class UserDataNode;
};

}  // namespace runtime
}  // namespace matxscript
