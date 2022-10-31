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
 * Trie builtin methods
 *****************************************************************************/

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(trie_update, update)
    .set_num_inputs(2)
    .set_num_inputs_max(3)
    .add_argument("self", "matx.Trie", "")
    .add_argument("w", "bytes_view|unicode_view|any_view", "")
    .add_argument("val", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(trie_prefix_search, prefix_search)
    .set_num_inputs(2)
    .set_num_inputs_max(3)
    .add_argument("self", "matx.Trie", "")
    .add_argument("w", "bytes_view|unicode_view|any_view", "")
    .add_argument("pos", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(trie_prefix_search_all, prefix_search_all)
    .set_num_inputs(2)
    .add_argument("self", "matx.Trie", "")
    .add_argument("w", "bytes_view|unicode_view|any_view", "")
    .add_argument("pos", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(trie_save, save)
    .set_num_inputs(2)
    .add_argument("self", "matx.Trie", "")
    .add_argument("file_path", "unicode_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(trie_load, load)
    .set_num_inputs(2)
    .add_argument("self", "matx.Trie", "")
    .add_argument("file_path", "unicode_view", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
