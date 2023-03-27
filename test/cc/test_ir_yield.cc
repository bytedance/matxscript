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
#include <matxscript/ir/function.h>
#include <matxscript/ir/module.h>
#include <matxscript/ir/prim_builtin.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>
#include <iostream>

namespace matxscript {
namespace ir {
using namespace runtime;

TEST(IR, Yield) {
  const auto* printer = ::matxscript::runtime::FunctionRegistry::Get("ir.AsText");
  const auto* build_module = ::matxscript::runtime::FunctionRegistry::Get("module.build.c");

  PrimVar n("n", DataType::Int(32));
  PrimVar i("i", DataType::Int(32));
  Array<Stmt> seq_stmt;
  For for_stmt(i,
               IntImm(DataType::Int(32), 0),
               n,
               IntImm(DataType::Int(32), 1),
               ForType::Serial,
               HLOYield(i));
  seq_stmt.push_back(for_stmt);
  SeqStmt body(seq_stmt);
  Array<BaseExpr> params{n};
  Function func(params, {}, body, ObjectType(), {});

  String ir_text = (*printer)({func}).As<String>();
  std::cout << ir_text << std::endl;

  func = WithAttr(std::move(func), attr::kGlobalSymbol, StringRef("test_generator"));

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
