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
#include <../src/codegen/codegen_c_host.h>
#include <gtest/gtest.h>
#include <matxscript/ir/expr.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/module.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/registry.h>
#include <iostream>

namespace matxscript {
namespace ir {
using namespace runtime;

TEST(IR, AllocaVarStmt) {
  DataType int_ty = DataType::Int(64);
  const auto* printer = ::matxscript::runtime::FunctionRegistry::Get("node.IRTextPrinter_Print");
  const auto* build_module = ::matxscript::runtime::FunctionRegistry::Get("module.build.c");

  AllocaVarStmt alloca_stmt("b", PrimType(int_ty), IntImm(int_ty, 0));
  AssignStmt assign_stmt(alloca_stmt->var, PrimExpr(10));
  ReturnStmt rt_stmt(alloca_stmt->var);

  Array<Stmt> seq_stmt;
  seq_stmt.push_back(alloca_stmt);
  seq_stmt.push_back(assign_stmt);
  seq_stmt.push_back(rt_stmt);
  SeqStmt body(seq_stmt);
  Array<PrimVar> params{};
  PrimFunc func(params, {}, body, PrimType(DataType::Int(32)));

  String ir_text = (*printer)({func, None}).As<String>();
  std::cout << ir_text << std::endl;

  func = WithAttr(std::move(func), attr::kGlobalSymbol, StringRef("test_alloca"));

  codegen::CodeGenCHost cg;
  cg.AddFunction(func);
  std::string code = cg.Finish();
  std::cout << code << std::endl;

  IRModule mod;
  mod->Add(func);
  ::matxscript::runtime::Module m = (*build_module)({mod}).As<Module>();
  std::cout << m->GetSource() << std::endl;
}

}  // namespace ir
}  // namespace matxscript
