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
#include <matxscript/runtime/func_registry_names_io.h>

#include <sstream>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

String GenerateFuncRegistryNames(const std::vector<String>& function_names) {
  std::stringstream ss;
  ss << std::to_string(function_names.size()) << '\0';
  for (auto& f : function_names) {
    ss << f << '\0';
  }

  return ss.str();
}

std::vector<string_view> ReadFuncRegistryNames(const char* names) {
  string_view num_str(names);
  uint64_t num = std::strtoull(num_str.data(), nullptr, 10);
  const char* reg_name_ptr = names + num_str.size() + 1;
  std::vector<string_view> result;

  result.reserve(num);
  // NOTE: reg_name_ptr starts at index 1 to skip num_funcs.
  for (size_t i = 0; i < num; ++i) {
    result.emplace_back(string_view(reg_name_ptr));
    reg_name_ptr += result.back().size() + 1;
  }
  return result;
}

int LookupFuncRegistryName(const char* names, string_view target) {
  size_t num = (unsigned char)names[0];
  const char* reg_name_ptr = names + 1;
  // NOTE: reg_name_ptr starts at index 1 to skip num_funcs.
  for (size_t i = 0; i < num; ++i) {
    string_view reg_name(reg_name_ptr);
    if (reg_name == target) {
      return i;
    }
    reg_name_ptr += reg_name.size() + 1;
  }
  return -1;
}

}  // namespace runtime
}  // namespace matxscript
