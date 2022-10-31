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

MATXSCRIPT_REGISTER_GLOBAL("runtime.RTValue_GetTypeCode").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[runtime.RTValue_GetTypeCode] Expect 1 arguments but get "
                             << args.size();
  return args[0].type_code();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.RTValue_Repr").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "RuntimeObjectRepr expect " << 1 << " arguments but get "
                            << args.size();
  std::stringstream os;
  os << args[0];
  return StringHelper::Decode(os.str());
});

}  // namespace runtime
}  // namespace matxscript
