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

#include <matxscript/pipeline/jit_object.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("runtime.JitObject_GetSelf").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[JitObject_GetSelf] Expect 1 arguments but get " << args.size();
  UserDataRef ud_ref = args[0].As<UserDataRef>();
  auto jit_obj_ptr = check_get_jit_object(ud_ref);
  return jit_obj_ptr->self();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.JitObject_GetFunction").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[JitObject_GetFunction] Expect 2 arguments but get "
                             << args.size();
  UserDataRef ud_ref = args[0].As<UserDataRef>();
  auto jit_obj_ptr = check_get_jit_object(ud_ref);
  MXCHECK(args[1].IsUnicode() || args[1].IsString());
  String name;
  if (args[1].IsUnicode()) {
    name = args[1].As<Unicode>().encode();
  } else {
    name = args[1].As<String>();
  }
  auto ret = jit_obj_ptr->GetFunction(name);
  return RTValue(new NativeFunction(ret.first), TypeIndex::kRuntimePackedFuncHandle);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.JitObject_GetFunctionSchema")
    .set_body([](PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 2) << "[JitObject_GetFunction] Expect 2 arguments but get "
                                 << args.size();
      UserDataRef ud_ref = args[0].As<UserDataRef>();
      auto jit_obj_ptr = check_get_jit_object(ud_ref);
      MXCHECK(args[1].IsUnicode() || args[1].IsString());
      String name;
      if (args[1].IsUnicode()) {
        name = args[1].As<Unicode>().encode();
      } else {
        name = args[1].As<String>();
      }
      auto ret = jit_obj_ptr->GetFunction(name);
      return ret.second->ToDict();
    });

}  // namespace runtime
}  // namespace matxscript
