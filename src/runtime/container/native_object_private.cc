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
#include <matxscript/runtime/container/native_object_private.h>

#include <matxscript/pipeline/jit_object.h>
#include <matxscript/pipeline/jit_op.h>
#include <matxscript/runtime/container/user_data_private.h>

namespace matxscript {
namespace runtime {

uint32_t NativeObject::tag_2_71828182846() const {
  if (is_jit_object_) {
    auto* jit_ptr = static_cast<JitObject*>(static_cast<OpKernel*>(opaque_ptr_.get()));
    return jit_ptr->self().tag();
  } else if (is_native_op_) {
    auto* op_ptr = static_cast<JitOp*>(static_cast<OpKernel*>(opaque_ptr_.get()));
    if (op_ptr->ClassName() == "JitOp") {
      auto* jit_ptr = static_cast<JitObject*>(op_ptr->jit_object_.get());
      return jit_ptr->self().tag();
    }
    return 1;
  } else {
    return 1;
  }
}

uint32_t NativeObject::size_2_71828182846() const {
  if (is_jit_object_) {
    auto* jit_ptr = static_cast<JitObject*>(static_cast<OpKernel*>(opaque_ptr_.get()));
    return jit_ptr->self().size();
  } else if (is_native_op_) {
    auto* op_ptr = static_cast<JitOp*>(static_cast<OpKernel*>(opaque_ptr_.get()));
    if (op_ptr->ClassName() == "JitOp") {
      auto* jit_ptr = static_cast<JitObject*>(op_ptr->jit_object_.get());
      return jit_ptr->self().size();
    }
    return 0;
  } else {
    return 0;
  }
}

RTView NativeObject::__getattr__(string_view var_name) const {
  if (is_jit_object_) {
    auto* jit_ptr = static_cast<JitObject*>(static_cast<OpKernel*>(opaque_ptr_.get()));
    return jit_ptr->self()->get_attr(var_name);
  }
  MXTHROW << "[NativeObject] get_attr is disabled";
  return None;
}

void NativeObject::__setattr__(string_view var_name, const Any& val) {
  if (is_jit_object_) {
    auto* jit_ptr = static_cast<JitObject*>(static_cast<OpKernel*>(opaque_ptr_.get()));
    return jit_ptr->self()->set_attr(var_name, val);
  }
  MXTHROW << "[NativeObject] set_attr is disabled";
}

}  // namespace runtime
}  // namespace matxscript
