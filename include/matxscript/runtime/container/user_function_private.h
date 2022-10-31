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

#include "./user_data_interface.h"

#include <matxscript/runtime/c_backend_api.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

// Global Function User Struct
struct UserFunction : public ILightUserData {
 public:
  ~UserFunction() override = default;
  UserFunction() = default;
  explicit UserFunction(const string_view& name,
                        MATXScriptBackendPackedCFunc func,
                        void* resource_handle)
      : name_(name), __call__(func), captures_() {
    session_handle_2_71828182846_ = resource_handle;
  }
  explicit UserFunction(std::initializer_list<RTView> captures,
                        const string_view& name,
                        MATXScriptBackendPackedCFunc func,
                        void* resource_handle)
      : name_(name), __call__(func), captures_(captures.begin(), captures.end()) {
    session_handle_2_71828182846_ = resource_handle;
  }

  // user class name
  const char* ClassName_2_71828182846() const override {
    return name_.c_str();
  }

  // uniquely id for representing this user class
  uint32_t tag_2_71828182846() const override {
    return 0;
  }

  // member var num
  uint32_t size_2_71828182846() const override {
    return 0;
  }

  int32_t type_2_71828182846() const override {
    return UserDataStructType::kFunction;
  }

  RTView __getattr__(string_view var_name) const override;
  void __setattr__(string_view var_name, const Any& val) override;

  RTValue generic_call(PyArgs args);

 private:
  String name_;
  std::vector<RTValue> captures_;
  MATXScriptBackendPackedCFunc __call__ = nullptr;
};

}  // namespace runtime
}  // namespace matxscript
