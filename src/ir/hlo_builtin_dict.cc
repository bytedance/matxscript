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
 * Dict builtin methods
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict___len__, size)
    .set_num_inputs(1)
    .add_argument("self", "dict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict___len__, size)
    .set_num_inputs(1)
    .add_argument("self", "FTDict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict___contains__, contains)
    .set_num_inputs(2)
    .add_argument("self", "dict", "")
    .add_argument("key", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict___contains__, contains)
    .set_num_inputs(2)
    .add_argument("self", "FTDict", "")
    .add_argument("key", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict___getitem__, get_item)
    .set_num_inputs(2)
    .add_argument("self", "dict", "")
    .add_argument("key", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict___getitem__, get_item)
    .set_num_inputs(2)
    .add_argument("self", "FTDict", "")
    .add_argument("key", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict___setitem__, set_item)
    .set_num_inputs(3)
    .add_argument("self", "dict", "")
    .add_argument("key", "Any", "")
    .add_argument("item", "Any", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict___setitem__, set_item)
    .set_num_inputs(3)
    .add_argument("self", "FTDict", "")
    .add_argument("key", "<template>", "")
    .add_argument("item", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_clear, clear)
    .set_num_inputs(1)
    .add_argument("self", "dict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_clear, clear)
    .set_num_inputs(1)
    .add_argument("self", "FTDict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_reserve, reserve)
    .set_num_inputs(2)
    .add_argument("self", "dict", "")
    .add_argument("new_size", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_reserve, reserve)
    .set_num_inputs(2)
    .add_argument("self", "FTDict", "")
    .add_argument("new_size", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_bucket_count, bucket_count)
    .set_num_inputs(1)
    .add_argument("self", "dict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_bucket_count, bucket_count)
    .set_num_inputs(1)
    .add_argument("self", "FTDict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_keys, key_iter)
    .set_num_inputs(1)
    .add_argument("self", "dict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_keys, key_iter)
    .set_num_inputs(1)
    .add_argument("self", "FTDict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_values, value_iter)
    .set_num_inputs(1)
    .add_argument("self", "dict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_values, value_iter)
    .set_num_inputs(1)
    .add_argument("self", "FTDict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_items, item_iter)
    .set_num_inputs(1)
    .add_argument("self", "dict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_items, item_iter)
    .set_num_inputs(1)
    .add_argument("self", "FTDict", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_get, get_default)
    .set_num_inputs(2)
    .set_num_inputs_max(3)
    .add_argument("self", "dict", "")
    .add_argument("key", "<template>", "")
    .add_argument("default_val", "Any", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_get, get_default)
    .set_num_inputs(2)
    .set_num_inputs_max(3)
    .add_argument("self", "FTDict", "")
    .add_argument("key", "<template>", "")
    .add_argument("default_val", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(dict_pop, pop)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "dict", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ft_dict_pop, pop)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "FTDict", "")
    .add_argument("args", "<template>", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
