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

#include <matxscript/ir/stmt_functor.h>

namespace matxscript {
namespace ir {

class AssignOptimizerFilter : public StmtExprVisitor {
 public:
  bool run(const BaseFunc& f) {
    if (!f->IsInstance<FunctionNode>()) {
      return false;
    }
    match_ = true;
    VisitStmt(runtime::Downcast<Function>(f)->body);
    return match_;
  }

  void VisitStmt(const Stmt& e) override {
    if (e->IsInstance<AllocaVarStmtNode>() || e->IsInstance<AssignStmtNode>() ||
        e->IsInstance<ReturnStmtNode>() || e->IsInstance<SeqStmtNode>()) {
    } else {
      match_ = false;
    }
    StmtExprVisitor::VisitStmt(e);
  }

 private:
  bool match_ = true;
};

class AssignOptimizerExprVisitor : public ExprVisitor {
 public:
  std::unordered_set<const void*> run(const BaseExpr& expr) {
    MXCHECK(func_args_);
    matched_.clear();
    this->VisitExpr(expr);
    return matched_;
  }

  void VisitExpr(const BaseExpr& expr) override {
    if (func_args_->count(expr.get())) {
      matched_.emplace(expr.get());
    }
    ExprVisitor::VisitExpr(expr);
  }

  std::unordered_set<const void*>* func_args_ = nullptr;
  std::unordered_set<const void*> matched_;
};

class AssignOptimizerLastUsedAnalysis : public StmtExprVisitor {
 public:
  std::unordered_map<const void*, const void*> run(const BaseFunc& f) {
    MXCHECK(func_args_);
    var_checker_.func_args_ = func_args_;
    AssignOptimizerFilter filter;
    if (!filter.run(f)) {
      return std::unordered_map<const void*, const void*>();
    } else {
      last_use_.clear();
      StmtExprVisitor::VisitStmt(runtime::Downcast<Function>(f)->body);
      return last_use_;
    }
  }

  void VisitStmt_(const AllocaVarStmtNode* op) override {
    std::unordered_set<const void*> var_matched = var_checker_.run(op->var);
    for (auto node_ptr : var_matched) {
      last_use_[node_ptr] = op;
    }
    std::unordered_set<const void*> val_matched = var_checker_.run(op->init_value);
    for (auto node_ptr : val_matched) {
      last_use_[node_ptr] = op;
    }
  }

  void VisitStmt_(const AssignStmtNode* op) override {
    std::unordered_set<const void*> lhs_matched = var_checker_.run(op->lhs);
    std::unordered_set<const void*> rhs_matched = var_checker_.run(op->rhs);
    for (auto node_ptr : lhs_matched) {
      last_use_[node_ptr] = op;
    }
    for (auto node_ptr : rhs_matched) {
      last_use_[node_ptr] = op;
    }
  }

  void VisitStmt_(const ReturnStmtNode* op) override {
    std::unordered_set<const void*> matched = var_checker_.run(op->value);
    for (auto node_ptr : matched) {
      last_use_[node_ptr] = op;
    }
  }

 public:
  AssignOptimizerExprVisitor var_checker_;
  std::unordered_map<const void*, const void*> last_use_;
  std::unordered_set<const void*>* func_args_ = nullptr;
};

class AssignOptimizerMutator : public StmtExprMutator {
 public:
  BaseFunc run(const BaseFunc& f) {
    func_args_.clear();
    last_use_stmt_.clear();
    auto func_args = f->GetParams();
    if (func_args.empty()) {
      return f;
    }
    for (auto& f_arg : func_args) {
      if (IsPrimType(f_arg->checked_type())) {
        continue;
      }
      func_args_.emplace(f_arg.get());
    }
    if (func_args_.empty()) {
      return f;
    }
    AssignOptimizerLastUsedAnalysis analysis;
    analysis.func_args_ = &func_args_;
    auto last_used = analysis.run(f);
    for (auto& last_stmt : last_used) {
      last_use_stmt_.emplace(last_stmt.second);
    }
    if (last_use_stmt_.empty()) {
      return f;
    } else {
      auto hlo_func = runtime::Downcast<Function>(f);
      auto body = StmtExprMutator::Mutate(hlo_func->body);
      if (body.same_as(hlo_func->body)) {
        return f;
      }
      auto new_func_node = CopyOnWrite(hlo_func.get());
      new_func_node->body = std::move(body);
      return BaseFunc(new_func_node);
    }
  }

  Stmt VisitStmt_(const AllocaVarStmtNode* op) override {
    if (op->init_value.defined() && last_use_stmt_.count(op) &&
        func_args_.count(op->init_value.get())) {
      auto new_stmt = CopyOnWrite(op);
      new_stmt->init_value = HLOMove(new_stmt->init_value);
      return Stmt(new_stmt);
    }
    return runtime::GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const AssignStmtNode* op) override {
    if (last_use_stmt_.count(op) && func_args_.count(op->rhs.get())) {
      auto new_stmt = CopyOnWrite(op);
      new_stmt->rhs = HLOMove(new_stmt->rhs);
      return Stmt(new_stmt);
    }
    return runtime::GetRef<Stmt>(op);
  }

 private:
  std::unordered_set<const void*> last_use_stmt_;
  std::unordered_set<const void*> func_args_;
};

}  // namespace ir
}  // namespace matxscript
