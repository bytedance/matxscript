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

#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetLoggingLevel").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[SetLoggingLevel] Expect 1 arguments but get " << args.size();
  auto logging_level = args[0].As<int64_t>();
  SetLoggingLevel(logging_level);
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.GetLoggingLevel").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 0) << "[GetLoggingLevel] Expect 0 arguments but get " << args.size();
  auto logging_level = GetLoggingLevel();
  return logging_level;
});

}  // namespace runtime
}  // namespace matxscript
