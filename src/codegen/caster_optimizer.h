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

class FuseContCasterOptimizer : public StmtExprMutator {
 public:
  BaseFunc run(const BaseFunc& func) {
    return runtime::Downcast<BaseFunc>(this->VisitStmt(func));
  }

  HLOExpr VisitExpr_(const HLOCastNode* op) override {
    const auto& op_type = RemoveReference(op->checked_type());
    auto raw_value = RemoveNestedCaster(op->value);
    if (raw_value.same_as(op->value)) {
      return runtime::GetRef<HLOExpr>(op);
    }
    return HLOCast(op->checked_type(), raw_value, op->span);
  }

  PrimExpr VisitExpr_(const HLOCastPrimNode* op) override {
    const auto& op_type = RemoveReference(op->checked_type());
    auto raw_value = RemoveNestedCaster(op->value);
    if (raw_value.same_as(op->value)) {
      return runtime::GetRef<PrimExpr>(op);
    }
    return HLOCastPrim(op->dtype, raw_value, op->span);
  }

 protected:
  static BaseExpr RemoveNestedCaster(BaseExpr e) {
    if (auto* cast_node = e.as<HLOCastNode>()) {
      return RemoveNestedCaster(cast_node->value);
    }
    if (auto* cast_node = e.as<HLOCastPrimNode>()) {
      return RemoveNestedCaster(cast_node->value);
    }
    return e;
  }
};

}  // namespace ir
}  // namespace matxscript
