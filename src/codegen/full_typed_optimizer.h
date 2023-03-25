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

class FullTypedOptimizerAnalysis : public StmtExprVisitor {
 public:
  FullTypedOptimizerAnalysis();
  std::unordered_map<const HLOVarNode*, Type> run(const BaseFunc& f) {
    this->result = {};

    if (!f->IsInstance<FunctionNode>()) {
      return std::unordered_map<const HLOVarNode*, Type>{};
    }
    StmtExprVisitor::VisitStmt_(f.as<FunctionNode>());
    std::unordered_map<const HLOVarNode*, Type> ret;
    for (auto& r : this->result) {
      ret.emplace(r.first, r.second.first);
    }
    return ret;
  }

  static const BaseExprNode* RemoveMove(const BaseExprNode* node);

 protected:
  bool IsListLiteral(const BaseExprNode* init);
  bool IsDictLiteral(const BaseExprNode* init);
  bool IsCandidate(const BaseExprNode* var, const BaseExprNode* init);
  Type InferNewVarType(const BaseExprNode* var, const BaseExprNode* init);

  void VisitStmt_(const AllocaVarStmtNode* op) override {
    if (auto* var_node = op->var.as<HLOVarNode>()) {
      const auto& var_type = RemoveReference(var_node->checked_type());
      if (IsCandidate(op->var.get(), op->init_value.get())) {
        auto ty = InferNewVarType(op->var.get(), op->init_value.get());
        if (ty.defined()) {
          result[var_node] = {ty, 0};
          return;
        }
      }
    }
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const HLOVarNode* e) override {
    auto var_iter = result.find(e);
    if (var_iter != result.end()) {
      if (var_iter->second.second <= 0) {
        result.erase(var_iter);
      } else {
        var_iter->second.second -= 1;
      }
    }
    return ExprVisitor::VisitExpr_(e);
  }

  void VisitExpr_(const CallNode* e) override {
    // check first arg is the var
    if (e->args.size() >= 1) {
      const BaseExprNode* self_node = RemoveMove(e->args[0].get());
      if (self_node->IsInstance<HLOVarNode>()) {
        auto* var_self_node = static_cast<const HLOVarNode*>(self_node);
        auto var_iter = result.find(var_self_node);
        if (var_iter != result.end()) {
          const auto& var_ty = std::get<0>(var_iter->second);
          // check list ops
          auto list_op_iter = supported_list_ops.find(e->op.get());
          if (list_op_iter != supported_list_ops.end() && (list_op_iter->second)(var_ty, e)) {
            var_iter->second.second += 1;
          }
          // check set ops
          auto set_op_iter = supported_set_ops.find(e->op.get());
          if (set_op_iter != supported_set_ops.end() && (set_op_iter->second)(var_ty, e)) {
            var_iter->second.second += 1;
          }
          // check dict ops
          auto dict_op_iter = supported_dict_ops.find(e->op.get());
          if (dict_op_iter != supported_dict_ops.end() && (dict_op_iter->second)(var_ty, e)) {
            var_iter->second.second += 1;
          }
        }
      }
    }
    return ExprVisitor::VisitExpr_(e);
  }

  // result
  std::unordered_map<const HLOVarNode*, std::pair<Type, int64_t>> result;

  typedef bool (*FuncCheckType)(const Type& var_type, const CallNode* call);
  std::unordered_map<const HLOExprNode*, FuncCheckType> supported_list_ops;
  std::unordered_map<const HLOExprNode*, FuncCheckType> supported_dict_ops;
  std::unordered_map<const HLOExprNode*, FuncCheckType> supported_set_ops;
};

class FullTypedOptimizerMutator : public StmtExprMutator {
 public:
  FullTypedOptimizerMutator();
  BaseFunc run(const BaseFunc& f) {
    FullTypedOptimizerAnalysis analysis;
    var_map_ = {};
    auto var_and_item_types = analysis.run(f);
    if (var_and_item_types.empty()) {
      return f;
    }
    for (auto& var_and_ty : var_and_item_types) {
      if (var_and_ty.second.defined()) {
        var_map_[var_and_ty.first] =
            HLOVar(var_and_ty.first->vid, var_and_ty.second, var_and_ty.first->span);
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

 protected:
  BaseExpr MutateLiteralValues(const BaseExpr& init, const Type& type);

  Stmt VisitStmt_(const AllocaVarStmtNode* op) final {
    if (auto* var_node = op->var.as<HLOVarNode>()) {
      auto var_iter = var_map_.find(var_node);
      if (var_iter != var_map_.end()) {
        auto new_init_value = MutateLiteralValues(op->init_value, var_iter->second->checked_type());
        if (new_init_value.defined()) {
          auto new_op = this->CopyOnWrite(op);
          new_op->var = var_iter->second;
          new_op->init_value = std::move(new_init_value);
          return Stmt(std::move(new_op));
        } else {
          // fallback
          var_map_.erase(var_iter);
        }
      }
    }
    return runtime::GetRef<Stmt>(op);
  }

  HLOExpr VisitExpr_(const HLOVarNode* op) final {
    auto var_iter = var_map_.find(op);
    if (var_iter != var_map_.end()) {
      return var_iter->second;
    }
    return runtime::GetRef<HLOExpr>(op);
  }

  HLOExpr VisitExpr_(const CallNode* op) override {
    // check first arg is the var
    auto op_iter = ops_mapping_.find(op->op.get());
    if (op_iter != ops_mapping_.end() && op->args.size() >= 1) {
      const BaseExprNode* self_node = FullTypedOptimizerAnalysis::RemoveMove(op->args[0].get());
      if (self_node->IsInstance<HLOVarNode>()) {
        auto* var_self_node = static_cast<const HLOVarNode*>(self_node);
        auto var_iter = var_map_.find(var_self_node);
        if (var_iter != var_map_.end()) {
          auto new_expr = ExprMutator::VisitExpr_(op);
          if (auto* new_call_node = new_expr.as<CallNode>()) {
            auto new_call_node_2 = this->CopyOnWrite(new_call_node);
            new_call_node_2->op = runtime::GetRef<HLOExpr>(op_iter->second);
            return HLOExpr(std::move(new_call_node_2));
          }
        }
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

 private:
  std::unordered_map<const HLOVarNode*, HLOVar> var_map_;
  std::unordered_map<const HLOExprNode*, const HLOExprNode*> ops_mapping_;
};

}  // namespace ir
}  // namespace matxscript
