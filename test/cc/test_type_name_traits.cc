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
#include <gtest/gtest.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/type_name_traits.h>

namespace matxscript {
namespace runtime {

namespace {
class MyTestNameTraitsV1 {
  int a = 0;
};
class MyTestNameTraitsV2 {
  int a = 0;
};
class MyTestNameTraitsV3 {
  int a = 0;
};
MATXSCRIPT_REGISTER_TYPE_NAME_TRAITS(MyTestNameTraitsV1);
}  // namespace
MATXSCRIPT_REGISTER_TYPE_NAME_TRAITS(MyTestNameTraitsV2);

TEST(TypeNameTraits, RegisterGet) {
  MATXSCRIPT_REGISTER_TYPE_NAME_TRAITS(MyTestNameTraitsV1);
  MATXSCRIPT_REGISTER_TYPE_NAME_TRAITS(MyTestNameTraitsV2);

  EXPECT_EQ(TypeNameTraits::Get<MyTestNameTraitsV1>(), "MyTestNameTraitsV1");
  EXPECT_EQ(TypeNameTraits::Get<MyTestNameTraitsV2>(), "MyTestNameTraitsV2");
  EXPECT_EQ(TypeNameTraits::Get<MyTestNameTraitsV3>(), "");
}

namespace {
class MyNativeDataExampleXXX {
 public:
  MyNativeDataExampleXXX() {
  }
  ~MyNativeDataExampleXXX() = default;

  String get_content() const {
    return "MyNativeDataExampleXXX";
  }
};
MATX_REGISTER_NATIVE_OBJECT(MyNativeDataExampleXXX)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 0) << "[MyNativeDataExampleXXX] Expect 0 arguments but get "
                                 << args.size();
      return std::make_shared<MyNativeDataExampleXXX>();
    })
    .RegisterFunction("get_content", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 0)
          << "[MyNativeDataExampleXXX][func: get_content] Expect 0 arguments but get "
          << args.size();
      return reinterpret_cast<MyNativeDataExampleXXX*>(self)->get_content();
    });
}  // namespace

TEST(TypeNameTraits, GetPipelineExample) {
  EXPECT_EQ(TypeNameTraits::Get<MyNativeDataExampleXXX>(), "MyNativeDataExampleXXX");
}

}  // namespace runtime
}  // namespace matxscript
