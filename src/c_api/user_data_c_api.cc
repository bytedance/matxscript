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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("runtime.UserData").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 0) << "[runtime.UserData] no need argument";
  return UserDataRef();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.UserData___getattr__").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2) << "[runtime.UserData] __getattr__ need 2 arguments, but get "
                            << args.size();
  auto ud = args[0].AsObjectView<UserDataRef>();
  auto name = args[1].As<string_view>();
  return ud.data().__getattr__(name);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.UserData___call__").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() >= 1) << "[runtime.UserData] __call__ need 1 or more arguments, but get "
                            << args.size();
  auto ud_view = args[0].AsObjectView<UserDataRef>();
  return ud_view.data().generic_call(PyArgs(args.begin() + 1, args.size() - 1));
});

}  // namespace runtime
}  // namespace matxscript
