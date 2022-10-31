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
#include <matxscript/ir/function.h>
#include <matxscript/ir/module.h>
#include <matxscript/ir/prim_builtin.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>
#include <iostream>

#include <gtest/gtest.h>

namespace matxscript {
namespace ir {
using namespace runtime;

TEST(IR, ForRange) {
  DataType f64_ty = DataType::Float(64);
  DataType i64_ty = DataType::Int(64);
  const auto* printer = ::matxscript::runtime::FunctionRegistry::Get("ir.AsText");

  PrimVar arg_ib("ib", i64_ty);
  AllocaVarStmt alloca_result("result", PrimType(f64_ty), FloatImm(f64_ty, 0.0));
  AssignStmt set_res(alloca_result->var,
                     add(HLOCastPrim(f64_ty, alloca_result->var), FloatImm(f64_ty, 1.0)));

  For loop(PrimVar("i"), IntImm(i64_ty, 0), arg_ib, IntImm(i64_ty, 1), ForType::Serial, set_res);
  Array<Stmt> seq_stmt;
  seq_stmt.push_back(alloca_result);
  seq_stmt.push_back(loop);
  seq_stmt.push_back(ReturnStmt(alloca_result->var));
  SeqStmt body(seq_stmt);
  Array<BaseExpr> params{arg_ib};
  Function func(params, {}, body, PrimType(DataType::Float(64)), {});

  String ir_text = (*printer)({func}).As<String>();
  std::cout << ir_text << std::endl;
}

}  // namespace ir
}  // namespace matxscript
