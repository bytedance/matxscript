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

class FuncArgsOptimizerLeftValueFilter : public ExprVisitor {
 public:
  void run(const BaseExpr& expr) {
    MXCHECK(func_args_);
    VisitExpr(expr);
  }

  void VisitExpr(const BaseExpr& expr) override {
    if (func_args_->count(expr.get())) {
      func_args_->erase(expr.get());
    }
    ExprVisitor::VisitExpr(expr);
  }

  std::unordered_set<const void*>* func_args_ = nullptr;
};

class FuncArgsOptimizerFilter : public StmtExprVisitor {
 public:
  std::unordered_set<const void*> run(const BaseFunc& f) {
    if (!f->IsInstance<FunctionNode>()) {
      return std::unordered_set<const void*>();
    }
    func_args_.clear();
    auto func_args = f->GetParams();
    if (func_args.empty()) {
      return std::unordered_set<const void*>();
    }
    for (auto& f_arg : func_args) {
      if (IsPrimType(f_arg->checked_type())) {
        continue;
      }
      if (IsNDArrayType(f_arg->checked_type())) {
        // TODO: fix ndarray const method
        continue;
      }
      func_args_.emplace(f_arg.get());
    }
    if (func_args_.empty()) {
      return std::unordered_set<const void*>();
    }
    VisitStmt(runtime::Downcast<Function>(f)->body);
    return func_args_;
  }

  void VisitStmt_(const AllocaVarStmtNode* op) override {
    FuncArgsOptimizerLeftValueFilter lhs_filter;
    lhs_filter.func_args_ = &func_args_;
    lhs_filter.run(op->var);
  }

  void VisitStmt_(const AssignStmtNode* op) override {
    if (func_args_.count(op->lhs.get())) {
      func_args_.erase(op->lhs.get());
    }
    // FuncArgsOptimizerLeftValueFilter lhs_filter;
    // lhs_filter.func_args_ = &func_args_;
    // lhs_filter.run(op->lhs);
  }

 private:
  std::unordered_set<const void*> func_args_;
};

class FuncArgsOptimizerMutator {
 public:
  BaseFunc run(const BaseFunc& f) {
    FuncArgsOptimizerFilter filter;
    auto mutator_args = filter.run(f);
    if (mutator_args.empty()) {
      return f;
    }
    auto hlo_func = runtime::Downcast<Function>(f);
    auto new_func_node = runtime::make_object<FunctionNode>(*hlo_func.get());
    runtime::Array<BaseExpr> new_args;
    runtime::Map<HLOVar, HLOExpr> value_map;
    for (auto& arg : new_func_node->params) {
      if (mutator_args.count(arg.get())) {
        if (auto* arg_ptr = arg.as<HLOVarNode>()) {
          auto new_var_node = runtime::make_object<HLOVarNode>(*arg_ptr);
          new_var_node->type_annotation = RefType(new_var_node->type_annotation);
          new_var_node->checked_type_ = RefType(new_var_node->checked_type_);
          new_args.push_back(HLOVar(new_var_node));
          value_map.Set(runtime::Downcast<HLOVar>(arg),
                        runtime::Downcast<HLOExpr>(new_args.back()));
        } else {
          new_args.push_back(arg);
        }
      } else {
        new_args.push_back(arg);
      }
    }
    new_func_node->params = new_args;
    new_func_node->body = Substitute(new_func_node->body, value_map);
    return BaseFunc(new_func_node);
  }
};

}  // namespace ir
}  // namespace matxscript
