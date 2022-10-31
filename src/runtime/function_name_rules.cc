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
#include <matxscript/runtime/function_name_rules.h>

#include <string>
#include <vector>

#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {
namespace FunctionNameRules {

static string_view class_method_sep("__F_");

String get_class_view_name(string_view class_name) {
  return String(class_name) + String("_SharedView");
}

String add_wrapper_suffix(string_view function_name) {
  return String::Concat(function_name, "_wrapper");
}

String add_packed_suffix(string_view function_name) {
  return String::Concat(function_name, "__c_api");
}

String add_class_prefix(string_view class_name, string_view method_name) {
  return String(class_name) + String("__F_") + String(method_name);
}

string_view remove_class_prefix(string_view class_name, string_view method_name) {
  MXCHECK(is_class_method(class_name, method_name));
  return method_name.substr(class_name.size() + class_method_sep.size());
}

bool is_class_method(string_view class_name, string_view method_name) {
  return method_name.size() > (class_name.size() + class_method_sep.size()) &&
         method_name.substr(0, class_name.size()) == class_name &&
         method_name.substr(class_name.size(), class_method_sep.size()) == class_method_sep;
}

}  // namespace FunctionNameRules
}  // namespace runtime
}  // namespace matxscript
