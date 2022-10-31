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
#include "move_optimizer.h"

namespace matxscript {
namespace ir {

MATXSCRIPT_REGISTER_GLOBAL("ir.MoveOptimizer_GetMoveVarAndLineno").set_body_typed([](BaseFunc f) {
  MoveOptimizerAnalysis analysis;
  MoveOptimizerCountVarUseCountAnalysis counter;
  auto& result = analysis.run(f);
  std::vector<runtime::Tuple> info;
  for (auto& var_last_usage : result) {
    if (1 == counter.run(runtime::GetRef<Stmt>(var_last_usage.second), var_last_usage.first)) {
      info.emplace_back(runtime::Tuple::dynamic(var_last_usage.first->name_hint(),
                                                var_last_usage.second->span->lineno));
    }
  }

  auto comp_func = [](const runtime::Tuple& lhs, const runtime::Tuple& rhs) {
    return lhs[1].AsNoCheck<int64_t>() <= rhs[1].AsNoCheck<int64_t>();
  };
  runtime::sort::pdqsort(info.begin(), info.end(), comp_func);
  return runtime::Tuple(info.begin(), info.end());
});

MATXSCRIPT_REGISTER_GLOBAL("ir.MoveOptimizerAnalysis").set_body_typed([](BaseFunc f) {
  MoveOptimizerAnalysis analysis;
  auto& result = analysis.run(f);
  runtime::Array<runtime::StringRef> pairs;
  for (auto& var_last_usage : result) {
    std::stringstream os;
    if (var_last_usage.first && var_last_usage.second) {
      os << "var: " << runtime::GetRef<BaseExpr>(var_last_usage.first)
         << ", stmt: " << runtime::GetRef<Stmt>(var_last_usage.second);
      pairs.push_back(runtime::StringRef(os.str()));
    }
  }
  return runtime::RTValue(pairs);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.MoveOptimizerMutator").set_body_typed([](BaseFunc f) {
  MoveOptimizerMutator optimizer;
  return runtime::RTValue(optimizer.run(f));
});

}  // namespace ir
}  // namespace matxscript
