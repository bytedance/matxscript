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
#include <matxscript/pipeline/jit_op.h>

#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(JitOp).SetThreadSafety(false);

void JitOp::Init() {
  main_func_name_ = GetAttr<String>("main_func_name");
  jit_object_name_ = GetAttr<String>("jit_object_name");
  jit_object_ = belong_to_->FindJitObject(jit_object_name_);
  MXCHECK(jit_object_ != nullptr);
  name_ = jit_object_->PyObjectName() + "_" + name_;
  sub_ops_ = {jit_object_};
  auto pf = jit_object_->GetFunction(main_func_name_);
  forward_func_ = pf.first;
  func_meta_ = pf.second;
  self_ = jit_object_->self();
  MXCHECK(forward_func_ != nullptr && func_meta_ != nullptr)
      << "[JitOp] function not found, name: " << main_func_name_;
  MXCHECK(self_.defined()) << "[JitOp] self is nullptr";
}

int JitOp::Bundle(string_view folder) {
  return 0;
}

RTValue JitOp::Process(PyArgs inputs) const {
  PyArgs real_inputs = inputs;
  std::unique_ptr<RTView[]> new_inputs;
  if (jit_object_->options_.is_class) {
    if (func_meta_->bound_self) {
      new_inputs.reset(new RTView[inputs.size() + 1]);
      new_inputs[0] = jit_object_->self_;
      for (size_t i = 0; i < inputs.size(); ++i) {
        new_inputs[i + 1] = inputs[i].As<RTView>();
      }
      real_inputs = PyArgs(new_inputs.get(), inputs.size() + 1);
    }
  } else {
    new_inputs.reset(new RTView[inputs.size() + 1]);
    // bundle session handler
    for (size_t i = 0; i < inputs.size(); ++i) {
      new_inputs[i] = inputs[i].As<RTView>();
    }
    new_inputs[inputs.size()] = RTView(belong_to_);
    real_inputs = PyArgs(new_inputs.get(), inputs.size() + 1);
  }
  return forward_func_(real_inputs);
}

RTValue JitOp::generic_call_attr(string_view func_name, PyArgs args) {
  return jit_object_->generic_call_attr(func_name, args);
}

}  // namespace runtime
}  // namespace matxscript
