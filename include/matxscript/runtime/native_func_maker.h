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

#include <initializer_list>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/user_data_ref.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

MATX_DLL UserDataRef make_native_function(string_view func_name);
MATX_DLL RTValue call_native_function(string_view func_name, PyArgs args);

template <typename... ARGS>
inline RTValue call_native_function(string_view func_name, ARGS&&... args) {
  std::initializer_list<RTView> args_init{std::forward<ARGS>(args)...};
  return call_native_function(func_name, PyArgs(args_init));
}

}  // namespace runtime
}  // namespace matxscript