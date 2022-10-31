// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the TypedNativeFunction is inspired by TVM.
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

#include <matxscript/runtime/py_args.h>

namespace matxscript {
namespace runtime {

using NativeMethod = std::function<RTValue(void* self, PyArgs args)>;
using NativeFunction = std::function<RTValue(PyArgs args)>;

template <>
MATXSCRIPT_ALWAYS_INLINE NativeFunction Any::As<NativeFunction>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimePackedFuncHandle);
  return NativeFunction(*reinterpret_cast<NativeFunction*>(value_.data.v_handle));
}

template <>
MATXSCRIPT_ALWAYS_INLINE NativeFunction Any::AsNoCheck<NativeFunction>() const {
  return NativeFunction(*reinterpret_cast<NativeFunction*>(value_.data.v_handle));
}

}  // namespace runtime
}  // namespace matxscript
