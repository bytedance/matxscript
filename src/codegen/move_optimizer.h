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

class MoveOptimizerAnalysis : public StmtExprVisitor {
  enum ScopeType {
    kFunction = 0,
    kIfElse = 1,
    kTryExcept = 1,
    kLoop = 2,
  };

 public:
  const std::unordered_map<const HLOVarNode*, const StmtNode*>& run(const BaseFunc& f) {
    this->yield_mode = false;
    this->result = {};
    this->symbols_ = {};
    this->scope_types_ = {};
    this->last_stmt_ = nullptr;

    if (!f->IsInstance<FunctionNode>()) {
      return this->result;
    }
    this->VisitStmt_(f.as<FunctionNode>());
    if (yield_mode) {
      this->result = {};
    }
    // filter nullptr
    std::unordered_map<const HLOVarNode*, const StmtNode*> final_res;
    for (auto& p : this->result) {
      if (p.first && p.second) {
        final_res.emplace(p.first, p.second);
      }
    }
    this->result = std::move(final_res);
    return this->result;
  }

 protected:
  bool CanMove(const BaseExprNode* e) {
    if (!e->IsInstance<HLOVarNode>()) {
      return false;
    }
    auto& type = RemoveReference(e->checked_type());
    if (type->IsInstance<PrimTypeNode>()) {
      return false;
    }
    if (auto* ty_node = type.as<StringTypeNode>()) {
      return !ty_node->is_view;
    }
    if (auto* ty_node = type.as<UnicodeTypeNode>()) {
      return !ty_node->is_view;
    }
    if (auto* ty_node = type.as<ObjectTypeNode>()) {
      return !ty_node->is_view;
    }
    return true;
  }

  void VisitStmt_(const FunctionNode* op) override {
    scope_types_.emplace_back(ScopeType::kFunction);
    symbols_.emplace_back();
    auto& current_symbols = symbols_.back();
    auto func_args = op->GetParams();
    for (auto& arg : func_args) {
      if (auto* arg_node = arg.as<HLOVarNode>()) {
        if (CanMove(arg_node)) {
          current_symbols[arg_node] = {};
        }
      }
    }
    this->VisitStmt(op->body);
    auto& current_symbols_2 = symbols_.back();
    for (auto& sym : current_symbols_2) {
      if (sym.second.size() == 1 && sym.second.begin()->second) {
        result[sym.first] = sym.second.begin()->second;
      }
    }
    symbols_.pop_back();
    scope_types_.pop_back();
  }

  void VisitExpr_(const HLOVarNode* e) override {
    // only the hlo expr can be moved
    auto& current_symbols = symbols_.back();
    auto sym_iter = current_symbols.find(e);
    if (sym_iter != current_symbols.end()) {
      sym_iter->second[scope_types_.back()] = last_stmt_;
    }
    for (int64_t i = int64_t(symbols_.size()) - 2; i >= 0; --i) {
      auto& current_symbols = symbols_[i];
      auto sym_iter = current_symbols.find(e);
      if (sym_iter != current_symbols.end()) {
        for (auto& scope_type_stmt : sym_iter->second) {
          scope_type_stmt.second = nullptr;
        }
      }
    }
  }

  void VisitStmt(const Stmt& e) override {
    auto* last_stmt = this->last_stmt_;
    last_stmt_ = e.get();
    StmtExprVisitor::VisitStmt(e);
    this->last_stmt_ = last_stmt;
  }

  void VisitStmt_(const AllocaVarStmtNode* op) override {
    if (auto* var_node = op->var.as<HLOVarNode>()) {
      if (CanMove(var_node)) {
        symbols_.back()[var_node] = {};
      }
    }
    return StmtExprVisitor::VisitExpr(op->init_value);
  }

  void VisitStmt_(const TryExceptNode* op) override {
    scope_types_.emplace_back(ScopeType::kTryExcept);
    symbols_.emplace_back();
    StmtExprVisitor::VisitStmt_(op);
    auto& current_symbols = symbols_.back();
    for (auto& sym : current_symbols) {
      if (sym.second.size() == 1 && sym.second.begin()->second) {
        result[sym.first] = sym.second.begin()->second;
      }
    }
    symbols_.pop_back();
    scope_types_.pop_back();
  }

  void VisitStmt_(const ExceptionHandlerNode* op) override {
    scope_types_.emplace_back(ScopeType::kTryExcept);
    symbols_.emplace_back();
    StmtExprVisitor::VisitStmt_(op);
    auto& current_symbols = symbols_.back();
    for (auto& sym : current_symbols) {
      if (sym.second.size() == 1 && sym.second.begin()->second) {
        result[sym.first] = sym.second.begin()->second;
      }
    }
    symbols_.pop_back();
    scope_types_.pop_back();
  }

  void VisitStmt_(const IfThenElseNode* op) override {
    scope_types_.emplace_back(ScopeType::kIfElse);
    symbols_.emplace_back();
    StmtExprVisitor::VisitStmt_(op);
    auto& current_symbols = symbols_.back();
    for (auto& sym : current_symbols) {
      if (sym.second.size() == 1 && sym.second.begin()->second) {
        result[sym.first] = sym.second.begin()->second;
      }
    }
    symbols_.pop_back();
    scope_types_.pop_back();
  }
  void VisitStmt_(const AutoForNode* op) override {
    scope_types_.emplace_back(ScopeType::kLoop);
    symbols_.emplace_back();
    auto& current_symbols = symbols_.back();
    for (auto& arg : op->loop_vars) {
      if (auto* arg_node = arg.as<HLOVarNode>()) {
        if (CanMove(arg_node)) {
          current_symbols[arg_node] = {};
        }
      }
    }
    this->VisitStmt(op->body);
    auto& current_symbols_2 = symbols_.back();
    for (auto& sym : current_symbols_2) {
      if (sym.second.size() == 1 && sym.second.begin()->second) {
        result[sym.first] = sym.second.begin()->second;
      }
    }
    symbols_.pop_back();
    scope_types_.pop_back();
  }

  void VisitStmt_(const ForNode* op) override {
    // prim for
    scope_types_.emplace_back(ScopeType::kLoop);
    symbols_.emplace_back();
    StmtExprVisitor::VisitStmt_(op);
    auto& current_symbols = symbols_.back();
    for (auto& sym : current_symbols) {
      if (sym.second.size() == 1 && sym.second.begin()->second) {
        result[sym.first] = sym.second.begin()->second;
      }
    }
    symbols_.pop_back();
    scope_types_.pop_back();
  }

  void VisitStmt_(const WhileNode* op) override {
    // prim for
    scope_types_.emplace_back(ScopeType::kLoop);
    symbols_.emplace_back();
    StmtExprVisitor::VisitStmt_(op);
    auto& current_symbols = symbols_.back();
    for (auto& sym : current_symbols) {
      if (sym.second.size() == 1 && sym.second.begin()->second) {
        result[sym.first] = sym.second.begin()->second;
      }
    }
    symbols_.pop_back();
    scope_types_.pop_back();
  }

  void VisitStmt_(const ReturnStmtNode* op) override {
    if (!op->value.as<HLOVarNode>()) {
      return StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const HLOYieldNode* op) override {
    yield_mode = true;
    return StmtExprVisitor::VisitStmt_(op);
  }

  const StmtNode* last_stmt_ = nullptr;
  // scope: var: branches_stmt
  std::vector<std::unordered_map<const HLOVarNode*, std::unordered_map<int, const StmtNode*>>>
      symbols_;
  std::vector<int> scope_types_;
  // result
  std::unordered_map<const HLOVarNode*, const StmtNode*> result;
  // bool yield_mode
  bool yield_mode = false;
};

class MoveOptimizerCountVarUseCountAnalysis : public StmtExprVisitor {
 public:
  int64_t run(const Stmt& e, const HLOVarNode* var) {
    this->count_ = 0;
    this->var_ = var;
    this->VisitStmt(e);
    int64_t count = this->count_;
    this->count_ = 0;
    this->var_ = nullptr;
    return count;
  }
  int64_t run(const BaseExpr& e, const HLOVarNode* var) {
    this->count_ = 0;
    this->var_ = var;
    this->VisitExpr(e);
    int64_t count = this->count_;
    this->count_ = 0;
    this->var_ = nullptr;
    return count;
  }

 protected:
  void VisitExpr_(const HLOVarNode* e) override {
    if (this->var_ == e) {
      this->count_++;
    }
  }

  const HLOVarNode* var_ = nullptr;
  int64_t count_ = 0;
};

class MoveOptimizerMutator : public StmtExprMutator {
 public:
  BaseFunc run(const BaseFunc& f) {
    MoveOptimizerAnalysis analysis;
    auto& define_and_usage_info = analysis.run(f);
    if (define_and_usage_info.empty()) {
      return f;
    }
    usage_and_define_.clear();
    for (auto& var_stmt : define_and_usage_info) {
      if (var_stmt.first->IsInstance<HLOVarNode>()) {
        usage_and_define_[var_stmt.second].emplace_back(var_stmt.first);
      }
    }
    auto hlo_func = runtime::Downcast<Function>(f);
    auto body = StmtExprMutator::Mutate(hlo_func->body);
    if (body.same_as(hlo_func->body)) {
      return f;
    }
    auto new_func_node = CopyOnWrite(hlo_func.get());
    new_func_node->body = std::move(body);
    return BaseFunc(new_func_node);
  }

  Stmt VisitStmt(const Stmt& op) override {
    auto stmt_iter = usage_and_define_.find(op.get());
    if (stmt_iter == usage_and_define_.end()) {
      return StmtMutator::VisitStmt(op);
    } else {
      // count use_count
      MoveOptimizerCountVarUseCountAnalysis counter;
      Stmt stmt = op;
      for (const HLOVarNode* var_node : stmt_iter->second) {
        if (auto* assign_node = stmt.as<AssignStmtNode>()) {
          if (counter.run(assign_node->lhs, var_node)) {
            // never move left value
            continue;
          }
        }
        auto count = counter.run(stmt, var_node);
        if (count == 1) {
          auto vmap = [&](const HLOVar& var) -> Optional<HLOExpr> {
            if (var.get() == var_node) {
              return HLOMove(var, var->span);
            }
            return Optional<HLOExpr>(nullptr);
          };
          stmt = Substitute(stmt, vmap);
        }
      }
      return stmt;
    }
    return op;
  }

  HLOExpr VisitExpr_(const HLOMoveNode* op) override {
    if (op->value->IsInstance<HLOVarNode>()) {
      return runtime::GetRef<HLOExpr>(op);
    }
    return ExprMutator::VisitExpr_(op);
  }

 private:
  std::unordered_map<const StmtNode*, std::vector<const HLOVarNode*>> usage_and_define_;
};

}  // namespace ir
}  // namespace matxscript
