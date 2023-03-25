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

class FuseContBinaryAddOptimizer : public StmtExprMutator {
 public:
  BaseFunc run(const BaseFunc& func) {
    return this->VisitStmt(func);
  }

  HLOExpr VisitExpr_(const HLOAddNode* op) override {
    const auto& op_type = RemoveReference(op->checked_type());
    if ((IsStringType(op_type) || IsUnicodeType(op_type))) {
      if (Check(op->a, op->b)) {
        auto lhs = this->VisitExpr(op->a);
        auto rhs = this->VisitExpr(op->b);
        return FlatContCallArgs(Call(op->checked_type(),
                                     IsStringType(op_type) ? builtin::str_fused_concat()
                                                           : builtin::unicode_fused_concat(),
                                     ir::Array<BaseExpr>{lhs, rhs},
                                     op->span)
                                    .get());
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

 protected:
  static HLOExpr FlatContCallArgs(const CallNode* op) {
    if (!op->op.same_as(builtin::str_fused_concat()) &&
        !op->op.same_as(builtin::unicode_fused_concat())) {
      return runtime::GetRef<HLOExpr>(op);
    }
    const auto* fused_op = &op->op;
    ir::Array<BaseExpr> call_args;
    std::function<void(const CallNode*)> func_flat_args;
    func_flat_args = [&](const CallNode* op) {
      if (op->op.same_as(*fused_op)) {
        for (auto& e : op->args) {
          if (auto* en = e.as<CallNode>()) {
            func_flat_args(en);
          } else {
            call_args.push_back(e);
          }
        }
      } else {
        call_args.push_back(runtime::GetRef<HLOExpr>(op));
      }
    };
    func_flat_args(op);
    if (call_args.size() <= 2) {
      return runtime::GetRef<HLOExpr>(op);
    }
    return Call(op->checked_type(), op->op, call_args, op->span, op->type_args);
  }
  static bool Check(const BaseExpr& lhs, const BaseExpr& rhs) {
    const auto& lhs_type = lhs->checked_type();
    const auto& rhs_type = rhs->checked_type();
    auto is_str_or_any = [](const Type& t) {
      auto& raw_ty = RemoveReference(t);
      return IsStringType(raw_ty) || IsObjectType(raw_ty);
    };
    auto is_unicode_or_any = [](const Type& t) {
      auto& raw_ty = RemoveReference(t);
      return IsUnicodeType(raw_ty) || IsObjectType(raw_ty);
    };

    auto matcher = [&](const Type& t1, const Type& t2) {
      auto& raw_ty1 = RemoveReference(t1);
      auto& raw_ty2 = RemoveReference(t2);
      return (IsStringType(raw_ty1) && is_str_or_any(raw_ty2)) ||
             (IsUnicodeType(raw_ty1) && is_unicode_or_any(raw_ty2));
    };
    return matcher(lhs_type, rhs_type) || matcher(rhs_type, lhs_type);
  }
};

}  // namespace ir
}  // namespace matxscript
