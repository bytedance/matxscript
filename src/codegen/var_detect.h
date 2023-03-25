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

#include <unordered_set>
#include <vector>

#include <matxscript/ir/stmt_functor.h>
#include <matxscript/runtime/bytes_hash.h>

namespace matxscript {
namespace ir {

class VarDetector : public StmtExprVisitor {
 public:
  void GetVars(const BaseFunc& f, std::vector<BaseExpr>& base_vars) {
    VisitStmt(f);
    base_vars = std::vector<BaseExpr>(vars_.begin(), vars_.end());
  }

 private:
  void VisitExpr_(const PrimVarNode* op) override {
    vars_.emplace(runtime::GetRef<PrimVar>(op));
  }
  void VisitExpr_(const HLOVarNode* op) override {
    vars_.emplace(runtime::GetRef<HLOVar>(op));
  }

 private:
  std::unordered_set<BaseExpr, ObjectHash, ObjectEqual> vars_;
};

class RemoveVarDefine : public StmtExprMutator {
 public:
  BaseFunc MutateFunc(BaseFunc& f) {
    return runtime::Downcast<BaseFunc>(StmtExprMutator::VisitStmt(f));
  }

 private:
  Stmt VisitStmt_(const AllocaVarStmtNode* op) override {
    return AssignStmt(op->var, op->init_value);
  }
  Stmt VisitStmt_(const ForNode* op) override {
    auto new_op = CopyOnWrite(op);
    new_op->yield_mode = true;
    return For(std::move(new_op));
  }
  Stmt VisitStmt_(const AutoForNode* op) override {
    auto new_op = CopyOnWrite(op);
    new_op->yield_mode = true;
    return AutoFor(std::move(new_op));
  }
};

BaseFunc SubstituteYieldFunctionVars(BaseFunc f, Map<BaseExpr, BaseExpr>& var_map) {
  auto make_new_var = [](BaseExpr var) -> BaseExpr {
    if (var->IsInstance<PrimVarNode>()) {
      PrimVar prim_var = runtime::Downcast<PrimVar>(var);
      StringRef raw_name = prim_var->name_hint;
      auto name_hash = runtime::BytesHash(raw_name.data(), raw_name.size());
      StringRef new_name = "__target_" + raw_name + std::to_string(name_hash);
      return PrimVar(new_name, prim_var->dtype);
    } else {
      HLOVar hlo_var = runtime::Downcast<HLOVar>(var);
      StringRef raw_name = hlo_var->name_hint();
      auto name_hash = runtime::BytesHash(raw_name.data(), raw_name.size());
      StringRef new_name = "__target_" + raw_name + std::to_string(name_hash);
      return HLOVar(new_name, hlo_var->type_annotation);
    }
  };

  // collect vars
  VarDetector vd;
  std::vector<BaseExpr> base_vars;
  vd.GetVars(f, base_vars);

  // make new var and build map
  for (auto& var : base_vars) {
    auto new_var = make_new_var(var);
    var_map.Set(var, new_var);
  }
  // remove var define
  auto f_no_alloca = RemoveVarDefine().MutateFunc(f);
  // replace var
  auto FuncSubstitute = [&var_map](const BaseExpr& var) -> Optional<BaseExpr> {
    auto it = var_map.find(var);
    if (it != var_map.end())
      return (*it).second;
    return Optional<BaseExpr>(nullptr);
  };
  f_no_alloca = runtime::Downcast<BaseFunc>(Substitute(f_no_alloca, FuncSubstitute));

  Array<Stmt> assgin_stmts;
  if (auto node_prim = f.as<PrimFuncNode>()) {
    for (auto& param : node_prim->params) {
      assgin_stmts.push_back(AssignStmt(var_map[param], param));
    }
    auto prim_func = runtime::Downcast<PrimFunc>(f_no_alloca);
    assgin_stmts.push_back(prim_func->body);
    auto new_node = prim_func.CopyOnWrite();
    new_node->body = SeqStmt(assgin_stmts);
    f_no_alloca = prim_func;
  } else {
    auto raw_hlo_func = runtime::Downcast<Function>(f);
    for (auto& param : raw_hlo_func->params) {
      assgin_stmts.push_back(AssignStmt(var_map[param], param));
    }
    auto hlo_func = runtime::Downcast<Function>(f_no_alloca);
    assgin_stmts.push_back(hlo_func->body);
    auto new_node = hlo_func.CopyOnWrite();
    new_node->body = SeqStmt(assgin_stmts);
    f_no_alloca = hlo_func;
  }
  return f_no_alloca;
}

}  // namespace ir
}  // namespace matxscript
