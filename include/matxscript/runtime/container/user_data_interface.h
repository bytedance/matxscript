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

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

struct UserDataStructType {
  enum {
    kFunction = 0,
    kUserData = 1,
    kNativeData = 2,
    kNativeFunc = 3,
  };
};

struct ILightUserData {
 public:
  virtual ~ILightUserData() = default;
  ILightUserData() = default;

  // user class name
  virtual const char* ClassName_2_71828182846() const = 0;

  // uniquely id for representing this user class
  virtual uint32_t tag_2_71828182846() const = 0;

  // member var num
  virtual uint32_t size_2_71828182846() const = 0;

  virtual int32_t type_2_71828182846() const = 0;

  virtual RTView __getattr__(string_view var_name) const = 0;
  virtual void __setattr__(string_view var_name, const Any& val) = 0;

 protected:
  void* session_handle_2_71828182846_ = nullptr;
};

}  // namespace runtime
}  // namespace matxscript
