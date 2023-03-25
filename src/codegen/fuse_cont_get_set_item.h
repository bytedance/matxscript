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
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <matxscript/ir/hlo_builtin.h>
#include <matxscript/ir/stmt_functor.h>

namespace matxscript {
namespace ir {

class FuseContAnyGetSetItemOptimizer : public StmtExprMutator {
 public:
  BaseFunc run(const BaseFunc& func) {
    return runtime::Downcast<BaseFunc>(this->VisitStmt(func));
  }

  HLOExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::object___setitem__())) {
      MXCHECK(op->args.size() == 3) << "internal error";
      Array<BaseExpr> keys;
      keys.push_back(op->args[1]);
      auto self = FlatContCallArgs(op->args[0], keys);
      if (!keys.empty()) {
        Array<BaseExpr> reverse_keys(keys.rbegin(), keys.rend());
        Array<BaseExpr> call_args;
        call_args.push_back(self);
        call_args.push_back(InitializerList(std::move(reverse_keys), op->span));
        call_args.push_back(op->args[2]);
        return Call(op->checked_type(),
                    builtin::object___fused_setitem__(),
                    call_args,
                    op->span,
                    op->type_args);
      }
    } else if (op->op.same_as(builtin::object___getitem__())) {
      MXCHECK(op->args.size() == 2) << "internal error";
      Array<BaseExpr> keys;
      keys.push_back(op->args[1]);
      auto self = FlatContCallArgs(op->args[0], keys);
      if (!keys.empty()) {
        Array<BaseExpr> reverse_keys(keys.rbegin(), keys.rend());
        Array<BaseExpr> call_args;
        call_args.push_back(self);
        call_args.push_back(InitializerList(std::move(reverse_keys), op->span));
        return Call(op->checked_type(),
                    builtin::object___fused_getitem__(),
                    call_args,
                    op->span,
                    op->type_args);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

 protected:
  static BaseExpr FlatContCallArgs(const BaseExpr& op, Array<BaseExpr>& keys) {
    auto* call_node = op.as<CallNode>();
    if (!call_node) {
      return op;
    }
    if (!call_node->op.same_as(builtin::object___getitem__())) {
      return op;
    }
    MXCHECK(call_node->args.size() == 2) << "internal error";
    keys.push_back(call_node->args[1]);
    return FlatContCallArgs(call_node->args[0], keys);
  }
};

}  // namespace ir
}  // namespace matxscript
