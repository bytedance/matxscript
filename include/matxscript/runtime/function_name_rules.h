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

#include <string>
#include <vector>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {

namespace FunctionNameRules {
MATX_DLL String get_class_view_name(string_view class_name);
MATX_DLL String add_wrapper_suffix(string_view function_name);
MATX_DLL String add_packed_suffix(string_view function_name);
MATX_DLL String add_class_prefix(string_view class_name, string_view method_name);
MATX_DLL string_view remove_class_prefix(string_view class_name, string_view method_name);
MATX_DLL bool is_class_method(string_view class_name, string_view method_name);
}  // namespace FunctionNameRules

}  // namespace runtime
}  // namespace matxscript
