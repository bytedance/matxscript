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

MATXSCRIPT_REGISTER_GLOBAL("runtime.Regex")
    .set_body_typed([](unicode_view pattern,
                       bool ignore_case,
                       bool dotall,
                       bool extended,
                       bool anchored,
                       bool ucp) {
      return Regex(pattern, ignore_case, dotall, extended, anchored, ucp);
    });

MATXSCRIPT_REGISTER_GLOBAL("runtime.RegexSplit").set_body_typed([](Regex regex_ref, RTView string) {
  return regex_ref.split(string);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.RegexReplace").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 3) << "runtime.RegexReplace expected 3 arguments but got " << args.size();
  Regex regex_ref = args[0].As<Regex>();
  return regex_ref.replace(args[1], args[2]);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.RegexMatch").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2 || args.size() == 3)
      << "runtime.RegexMatch expected 2 or 3 arguments but got " << args.size();
  Regex regex_ref = args[0].As<Regex>();
  if (args.size() == 2) {
    return regex_ref.match(args[1], 0);
  } else {
    return regex_ref.match(args[1], args[2].As<int64_t>());
  }
});

}  // namespace runtime
}  // namespace matxscript
