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
#include <matxscript/runtime/typed_native_function.h>
#include <matxscript/runtime/variadic_traits.h>

namespace matxscript {
namespace runtime {

template <typename FLambda, typename... DefaultArgs>
NativeFunction static set_generic_function(FLambda f, DefaultArgs&&... defaults) {
  using FType = typename variadic_details::function_signature<FLambda>::type;
  TypedNativeFunction<FType> ft(std::move(f));
  ft.SetDefaultArgs(std::forward<DefaultArgs>(defaults)...);
  return ft.packed();
}

template <typename FLambda, typename... DefaultArgs>
NativeMethod static set_class_member(FLambda f, DefaultArgs&&... defaults) {
  using FType = typename native_method_details::function_signature<FLambda>::type;
  TypedNativeFunction<FType> ft(std::move(f));
  ft.SetDefaultArgs(std::forward<DefaultArgs>(defaults)...);
  return ft.packed();
}

TEST(TypedNativeFunction, call_function) {
  auto add_one = [](int x) -> int { return x + 1; };
  TypedNativeFunction<int(int)> foo(add_one);
  int y = foo(1);
  EXPECT_EQ(y, 2);
  int y2 = foo.packed()({1}).As<int>();
  EXPECT_EQ(y2, 2);

  auto func = set_generic_function([](int x, int y) -> int { return x + y; }, 0, 1);
  auto r = func({});
  EXPECT_EQ(r, 1);
  r = func({1});
  EXPECT_EQ(r, 2);
  r = func({2, 3});
  EXPECT_EQ(r, 5);
}

TEST(TypedNativeFunction, call_class_method) {
  struct MyFoo {
    int b = 1;
  };
  auto add_one1 = [](void* self) -> int { return 1 + reinterpret_cast<MyFoo*>(self)->b; };
  TypedNativeFunction<int(void* self)> foo1(add_one1);
  auto add_one = [](void* self, int x) -> int { return x + reinterpret_cast<MyFoo*>(self)->b; };
  TypedNativeFunction<int(void* self, int)> foo(add_one);
  MyFoo foo_obj;
  int y = foo(&foo_obj, 1);
  EXPECT_EQ(y, 2);
  int y2 = foo.packed()(&foo_obj, {1}).As<int>();
  EXPECT_EQ(y2, 2);
}

TEST(TypedNativeFunction, call_method_default) {
  struct MyFoo {
    int b = 1;
  };
  auto add_one = [](void* self, int x) -> int { return x + reinterpret_cast<MyFoo*>(self)->b; };
  MyFoo foo_obj;
  auto func = set_class_member(add_one, 1);
  auto r = func(&foo_obj, {});
  std::cout << r << std::endl;
  r = func(&foo_obj, {2});
  std::cout << r << std::endl;
}

TEST(TypedNativeFunction, binding_defaults) {
  auto func = [](int x, int y) -> int { return x + y; };
  auto funcs = native_function_details::gen_lambdas_with_defaults<int(int, int)>(func, 0, 1);
  auto r = funcs[0]({1});
  EXPECT_EQ(r, 2);
}

}  // namespace runtime
}  // namespace matxscript
