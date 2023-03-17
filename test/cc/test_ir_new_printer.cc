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
#include <iostream>

#include <gtest/gtest.h>

#include <matxscript/ir/prim_builtin.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/ir/printer/text_printer.h>
#include <matxscript/ir/stmt.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

TEST(IRTextPrinter, PrintAllocaVar) {
  PrimExpr a(3);
  PrimExpr b(4);

  PrimAdd c(a, b);
  PrimMul d(c, a);

  Bool cond(true);
  PrimCall if_expr(d.dtype(), builtin::if_then_else(), {cond, d, c});

  PrimCast cast_expr(runtime::DataType::Int(32), if_expr);

  runtime::DataType int_ty = runtime::DataType::Int(64);
  AllocaVarStmt alloca_stmt("b", PrimType(int_ty), cast_expr);

  auto ir_text = printer::IRTextPrinter::Print(alloca_stmt, printer::PrinterConfig());
  // b: "int" = 0
  std::cout << ir_text << std::endl;
}

}  // namespace ir
}  // namespace matxscript
