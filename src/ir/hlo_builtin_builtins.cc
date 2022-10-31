// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the expressions is inspired by TVM.
 *
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
#include <matxscript/ir/hlo_builtin.h>
#include <matxscript/ir/hlo_expr.h>
#include "./hlo_builtin_macros.h"

namespace matxscript {
namespace ir {
namespace builtin {

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(builtins, print)
    .set_num_inputs(3)
    .set_num_inputs_max(-1);

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(builtins, ord)
    .set_num_inputs(1)
    .add_argument("c", "bytes_view|unicode_view|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(builtins, chr)
    .set_num_inputs(1)
    .add_argument("i", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(builtins, sorted)
    .set_num_inputs(1)
    .set_num_inputs_max(3)
    .add_argument("iterable", "List|Tuple|Any|any_view", "")
    .add_argument("key", "any_view|Any", "")
    .add_argument("reverse", "bool", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(builtins, unpack)
    .set_num_inputs(1)
    .add_argument("container", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC(hlo_if_then_else)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

// if_then_else
BaseExpr hlo_if_then_else(BaseExpr cond, BaseExpr true_value, BaseExpr false_value, Span span) {
  if (const auto* op = cond.as<IntImmNode>()) {
    if (op->value != 0) {
      return true_value;
    } else {
      return false_value;
    }
  }
  Type ret_type = ObjectType();
  if (true_value->checked_type() == false_value->checked_type()) {
    ret_type = true_value->checked_type();
  }
  return Call(std::move(ret_type),
              builtin::hlo_if_then_else(),
              {std::move(cond), std::move(true_value), std::move(false_value)},
              std::move(span));
}

MATXSCRIPT_REGISTER_GLOBAL("ir._HLOOpIfThenElse")
    .set_body_typed(
        [](BaseExpr cond, BaseExpr true_value, BaseExpr false_value, Span span = Span()) {
          return hlo_if_then_else(
              std::move(cond), std::move(true_value), std::move(false_value), std::move(span));
        });

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(builtins, isinstance).set_num_inputs(1);

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
