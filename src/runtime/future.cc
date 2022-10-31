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
#include <matxscript/pipeline/global_unique_index.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/file_util.h>
#include <matxscript/runtime/native_object_maker.h>
#include <matxscript/runtime/native_object_registry.h>

#include <matxscript/runtime/future_wrap.h>

namespace matxscript {
namespace runtime {

RTValue Future::get() const {
  return body_();
}

void Future::set_body(std::function<RTValue()> body) {
  this->body_ = std::move(body);
}

MATX_REGISTER_NATIVE_OBJECT(Future)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_LT(args.size(), 2) << "[Lazy Construction] Expect 0 or 1 arguments but get "
                                 << args.size();
      if (args.size() == 0) {
        return std::make_shared<Future>();
      } else {
        UserDataRef op = args[0].As<UserDataRef>();
        auto lazy = std::make_shared<Future>();
        lazy->set_body([op]() { return op.call(); });
        return lazy;
      }
    })
    .RegisterFunction("get",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 0)
                            << "[Future][func: get] Expect 0 arguments but get " << args.size();
                        return reinterpret_cast<Future*>(self)->get();
                      })
    .RegisterFunction("__call__", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 0) << "[Future][func: get] Expect 0 arguments but get "
                                 << args.size();
      return reinterpret_cast<Future*>(self)->get();
    });

UserDataRef Future::make_future_udref(std::function<RTValue()> body) {
  auto udref = make_native_userdata("Future", {});
  Future* future =
      static_cast<Future*>((static_cast<NativeObject*>(udref.ud_ptr())->opaque_ptr_).get());
  future->set_body(std::move(body));
  return udref;
}

}  // namespace runtime
}  // namespace matxscript
