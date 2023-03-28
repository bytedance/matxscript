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
#include <matxscript/ir/tuple_expr.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>
#include <iostream>

namespace matxscript {
namespace ir {
using namespace runtime;

TEST(IR, ListContainer) {
  const auto* printer = ::matxscript::runtime::FunctionRegistry::Get("node.IRTextPrinter_Print");
  ListType list_ty;

  auto list_append_op = Op::Get("ir.list_append");
  Constructor list_constructor(ListType(), "List", {}, GlobalTypeVar());
  Call call_list_init(list_ty, list_constructor, {InitializerList({StringImm("hello")})});

  AllocaVarStmt alloca_var("result", list_ty, call_list_init);
  Call call_list_append(list_ty, list_append_op, {alloca_var->var, StringImm("test")});
  ExprStmt es1(call_list_append);
  ReturnStmt ret_stmt(alloca_var->var);

  Array<Stmt> seqs;
  seqs.push_back(alloca_var);
  seqs.push_back(es1);
  seqs.push_back(ret_stmt);

  Function func({}, {}, SeqStmt(seqs), list_ty, {});
  String ir_text = (*printer)({func, None}).As<String>();

  func = WithAttr(std::move(func), attr::kGlobalSymbol, StringRef("return_list"));
  std::cout << ir_text << std::endl;

  codegen::CodeGenCHost cg;
  cg.AddFunction(func);
  std::string code = cg.Finish();
  std::cout << code << std::endl;
}

}  // namespace ir
}  // namespace matxscript
