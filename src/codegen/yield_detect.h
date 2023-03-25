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

#include <vector>

#include <matxscript/ir/stmt_functor.h>

namespace matxscript {
namespace ir {

class YieldDetector : public StmtExprVisitor {
 public:
  const std::vector<HLOYield>& GetYields(const BaseFunc& f) {
    yields_.clear();
    VisitStmt(f);
    return yields_;
  }

  void VisitStmt_(const HLOYieldNode* op) override {
    yields_.push_back(runtime::GetRef<HLOYield>(op));
  }

 private:
  std::vector<HLOYield> yields_;
};

class YieldLabelMutator : public StmtExprMutator {
 public:
  BaseFunc MutateFunc(const BaseFunc& f) {
    return runtime::Downcast<BaseFunc>(StmtExprMutator::VisitStmt(f));
  }

  Stmt VisitStmt_(const HLOYieldNode* op) override {
    auto node = CopyOnWrite(op);
    node->label = IntImm(runtime::DataType::Int(64), yield_index++);
    return HLOYield(node);
  }

 private:
  int64_t yield_index = 1;
};

}  // namespace ir
}  // namespace matxscript
