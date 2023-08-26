// Copyright 2023 ByteDance Ltd. and/or its affiliates.
//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "matxscript/runtime/mlir/func_loader.h"
#include "matxscript/runtime/py_args.h"
namespace matxscript {
namespace runtime {
namespace mlir {

void* load_func(const std::string& func_name, const std::string& share_lib_path) {
  if (lib_func_map.find(share_lib_path) == lib_func_map.end()) {
    void* lib = dlopen(share_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    MXCHECK(lib != nullptr) << "shared lib is not found in " << share_lib_path;
    std::pair<void*, std::unordered_map<std::string, void*>> value(lib, {});
    lib_func_map.emplace(share_lib_path, std::move(value));
  }
  auto& value = lib_func_map[share_lib_path];
  void* lib = value.first;
  auto& func_map = value.second;
  if (func_map.find(func_name) == func_map.end()) {
    auto func_ptr = dlsym(lib, (func_name.c_str()));
    MXCHECK(func_ptr != nullptr) << func_name << " is not found";
    func_map.emplace(func_name, func_ptr);
  }
  return func_map[func_name];
}

}  // namespace mlir
}  // namespace runtime
}  // namespace matxscript