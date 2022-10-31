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

/******************************************************************************
 * String(bytes) unbound methods
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, __len__)
    .set_num_inputs(1)
    .add_argument("self", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, __contains__)
    .set_num_inputs(2)
    .add_argument("self", "bytes_view", "")
    .add_argument("key", "int|bytes_view|Any|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, __getitem__)
    .set_num_inputs(2)
    .add_argument("self", "bytes_view", "")
    .add_argument("pos", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, __getslice__)
    .set_num_inputs(3)
    .set_num_inputs_max(4)
    .add_argument("self", "bytes_view", "")
    .add_argument("b", "int", "")
    .add_argument("e", "int", "")
    .add_argument("step", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, lower)
    .set_num_inputs(1)
    .add_argument("self", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, upper)
    .set_num_inputs(1)
    .add_argument("self", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, isdigit)
    .set_num_inputs(1)
    .add_argument("self", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, isalpha)
    .set_num_inputs(1)
    .add_argument("self", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, decode)
    .set_num_inputs(1)
    .add_argument("self", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, split)
    .set_num_inputs(1)
    .set_num_inputs_max(3)
    .add_argument("self", "bytes_view", "")
    .add_argument("sep", "bytes_view|Any|any_view", "")
    .add_argument("maxsplit", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, split_ft)
    .set_num_inputs(1)
    .set_num_inputs_max(3)
    .add_argument("self", "bytes_view", "")
    .add_argument("sep", "bytes_view|Any|any_view", "")
    .add_argument("maxsplit", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, join)
    .set_num_inputs(2)
    .add_argument("self", "bytes_view", "")
    .add_argument("iterable", "list|FTList[bytes]|Any|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, replace)
    .set_num_inputs(3)
    .set_num_inputs_max(4)
    .add_argument("self", "bytes_view", "")
    .add_argument("old_s", "bytes_view", "")
    .add_argument("new_s", "bytes_view", "")
    .add_argument("count", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, startswith)
    .set_num_inputs(2)
    .set_num_inputs_max(4)
    .add_argument("self", "bytes_view", "")
    .add_argument("prefix", "bytes_view|Tuple|Any|any_view", "")
    .add_argument("start", "int", "")
    .add_argument("end", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, endswith)
    .set_num_inputs(2)
    .set_num_inputs_max(4)
    .add_argument("self", "bytes_view", "")
    .add_argument("suffix", "bytes_view|Tuple|Any|any_view", "")
    .add_argument("start", "int", "")
    .add_argument("end", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, lstrip)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "bytes_view", "")
    .add_argument("chars", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, rstrip)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "bytes_view", "")
    .add_argument("chars", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, strip)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "bytes_view", "")
    .add_argument("chars", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, count)
    .set_num_inputs(2)
    .set_num_inputs_max(4)
    .add_argument("self", "bytes_view", "")
    .add_argument("sub", "bytes_view", "")
    .add_argument("start", "int", "")
    .add_argument("end", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, repeat)
    .set_num_inputs(2)
    .add_argument("self", "bytes_view", "")
    .add_argument("times", "int", "");

/******************************************************************************
 * String(bytes) fused ops
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(str, fused_concat)
    .set_num_inputs(2)
    .set_num_inputs_max(-1)
    .add_argument("s1", "bytes_view", "")
    .add_argument("s2", "bytes_view", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
