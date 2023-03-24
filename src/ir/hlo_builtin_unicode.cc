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
#include "./hlo_builtin_macros.h"

namespace matxscript {
namespace ir {
namespace builtin {

#define MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(Prefix, OpName)                              \
  MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC(Prefix##_##OpName)                                     \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))          \
      .set_attr<TGlobalSymbol>("TGlobalSymbol", MATXSCRIPT_AS_STR(kernel_##Prefix##_##OpName)) \
      .set_attr<TPrinterMethodSymbol>("TPrinterMethodSymbol", #OpName)

/******************************************************************************
 * Unicode(Python3 str) unbound methods
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, __len__)
    .set_num_inputs(1)
    .add_argument("self", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, __contains__)
    .set_num_inputs(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("key", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, __getitem__)
    .set_num_inputs(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("pos", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, __getslice__)
    .set_num_inputs(3)
    .set_num_inputs_max(4)
    .add_argument("self", "unicode_view", "")
    .add_argument("b", "int", "")
    .add_argument("e", "int", "")
    .add_argument("step", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, find)
    .set_num_inputs(2)
    .set_num_inputs_max(4)
    .add_argument("self", "unicode_view", "")
    .add_argument("sub", "unicode_view", "")
    .add_argument("start", "int", "")
    .add_argument("end", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, encode)
    .set_num_inputs(1)
    .add_argument("self", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, lower)
    .set_num_inputs(1)
    .add_argument("self", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, upper)
    .set_num_inputs(1)
    .add_argument("self", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, isdigit)
    .set_num_inputs(1)
    .add_argument("self", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, isalpha)
    .set_num_inputs(1)
    .add_argument("self", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, split)
    .set_num_inputs(1)
    .set_num_inputs_max(3)
    .add_argument("self", "unicode_view", "")
    .add_argument("sep", "unicode_view|Any|any_view", "")
    .add_argument("maxsplit", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, split_ft)
    .set_num_inputs(1)
    .set_num_inputs_max(3)
    .add_argument("self", "unicode_view", "")
    .add_argument("sep", "unicode_view|Any|any_view", "")
    .add_argument("maxsplit", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, join)
    .set_num_inputs(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("iterable", "list|FTList[str]|Any|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, replace)
    .set_num_inputs(3)
    .set_num_inputs_max(4)
    .add_argument("self", "unicode_view", "")
    .add_argument("old_s", "unicode_view", "")
    .add_argument("new_s", "unicode_view", "")
    .add_argument("count", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, startswith)
    .set_num_inputs(2)
    .set_num_inputs_max(4)
    .add_argument("self", "unicode_view", "")
    .add_argument("prefix", "unicode_view|Tuple|Any|any_view", "")
    .add_argument("start", "int", "")
    .add_argument("end", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, endswith)
    .set_num_inputs(2)
    .set_num_inputs_max(4)
    .add_argument("self", "unicode_view", "")
    .add_argument("suffix", "unicode_view|Tuple|Any|any_view", "")
    .add_argument("start", "int", "")
    .add_argument("end", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, lstrip)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("chars", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, rstrip)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("chars", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, strip)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("chars", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, count)
    .set_num_inputs(2)
    .set_num_inputs_max(4)
    .add_argument("self", "unicode_view", "")
    .add_argument("sub", "unicode_view", "")
    .add_argument("start", "int", "")
    .add_argument("end", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, format)
    .set_num_inputs(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, repeat)
    .set_num_inputs(2)
    .add_argument("self", "unicode_view", "")
    .add_argument("times", "int", "");

/******************************************************************************
 * Unicode(Python3 str) fused ops
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_UNICODE_FUNCTION(unicode, fused_concat)
    .set_num_inputs(2)
    .set_num_inputs_max(-1)
    .add_argument("s1", "unicode_view", "")
    .add_argument("s2", "unicode_view", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
