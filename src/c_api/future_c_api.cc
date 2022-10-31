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
#include <matxscript/runtime/future_wrap.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("runtime.Future").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[Future][func: Future] Expect 1 arguments but get " << args.size();
  UserDataRef op = args[0].AsObjectRef<UserDataRef>();
  return Future::make_future_udref([op]() -> RTValue { return op.call(); });
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Future_Get").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[Future][func: get] Expect 1 arguments but get " << args.size();
  UserDataRef udref = args[0].AsObjectRef<UserDataRef>();
  Future* future =
      static_cast<Future*>((static_cast<NativeObject*>(udref.ud_ptr())->opaque_ptr_).get());
  return future->get();
});

}  // namespace runtime
}  // namespace matxscript
