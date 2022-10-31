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

/******************************************************************************
 * Tuple container
 *****************************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("runtime.GetTupleSize").set_body([](PyArgs args) -> RTValue {
  const auto& tup = args[0].As<Tuple>();
  return static_cast<int64_t>(tup.size());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.GetTupleFields").set_body([](PyArgs args) -> RTValue {
  int64_t idx = args[1].As<int64_t>();
  const auto& tup = args[0].As<Tuple>();
  MXCHECK_LT(idx, tup.size());
  return tup[idx];
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Tuple").set_body([](PyArgs args) -> RTValue {
  std::vector<RTValue> fields;
  fields.reserve(args.size());
  for (auto i = 0; i < args.size(); ++i) {
    fields.push_back(args[i].As<RTValue>());
  }
  return Tuple(std::make_move_iterator(fields.begin()), std::make_move_iterator(fields.end()));
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.TupleEqual").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "tuple.__eq__ expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  const auto& other = args[1];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeTuple: {
      if (other.IsObjectRef<Tuple>()) {
        return self.AsObjectRefNoCheck<Tuple>() == other.AsObjectRefNoCheck<Tuple>();
      }
    } break;
    default: {
      MXTHROW << "expect 'tuple' but get '" << self.type_name();
    } break;
  }
  return false;
});

}  // namespace runtime
}  // namespace matxscript
