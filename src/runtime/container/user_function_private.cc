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
#include <matxscript/runtime/container/user_function_private.h>

namespace matxscript {
namespace runtime {

RTView UserFunction::__getattr__(string_view var_name) const {
  MXCHECK(false) << "[UserFunction] get_attr is disabled";
  return None;
}

void UserFunction::__setattr__(string_view var_name, const Any& val) {
  MXCHECK(false) << "[UserFunction] set_attr is disabled";
}

RTValue UserFunction::generic_call(PyArgs args) {
  const auto kNumArgs = args.size() + captures_.size();
  const auto kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  auto capture_size = captures_.size();
  MATXScriptAny values[kArraySize];
  for (size_t i = 0; i < capture_size; ++i) {
    values[i] = captures_[i].value();
  }
  for (size_t i = capture_size; i < kNumArgs; ++i) {
    values[i] = args[i - capture_size].value();
  }
  MATXScriptAny out_ret_value;
  __call__(values, kNumArgs, &out_ret_value, session_handle_2_71828182846_);
  return RTValue::MoveFromCHost(&out_ret_value);
}

}  // namespace runtime
}  // namespace matxscript
