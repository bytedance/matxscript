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
 * Set builtin methods
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, __len__, size)
    .set_num_inputs(1)
    .add_argument("self", "set", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, __len__, size)
    .set_num_inputs(1)
    .add_argument("self", "FTSet", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, __contains__, contains)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("key", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, __contains__, contains)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("key", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, add, add)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("item", "Any", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, add, add)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("item", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, clear, clear)
    .set_num_inputs(1)
    .add_argument("self", "set", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, clear, clear)
    .set_num_inputs(1)
    .add_argument("self", "FTSet", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, reserve, reserve)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("new_size", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, reserve, reserve)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("new_size", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, bucket_count, bucket_count)
    .set_num_inputs(1)
    .add_argument("self", "set", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, bucket_count, bucket_count)
    .set_num_inputs(1)
    .add_argument("self", "FTSet", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, difference, difference)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, difference, difference)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, difference_update, difference_update)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, difference_update, difference_update)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, update, update)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, update, update)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, union, set_union)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, union, set_union)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(set, discard, discard)
    .set_num_inputs(2)
    .add_argument("self", "set", "")
    .add_argument("rt_value", "Any|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(ft_set, discard, discard)
    .set_num_inputs(2)
    .add_argument("self", "FTSet", "")
    .add_argument("rt_value", "<template>", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
