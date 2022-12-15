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

// UserData dispatch
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __dispatch__)
    .set_num_inputs(3)
    .add_argument("self", "any_view", "")
    .add_argument("func_name", "bytes_view", "")
    .add_argument("args", "*args", "");

/******************************************************************************
 * python object data model special method names
 *
 * object.__len__(self)
 * object.__getitem__(self, key)
 * object.__setitem__(self, key, value)
 * object.__delitem__(self, key)
 * object.__contains__(self, item)
 * object.__hash__(self)
 * object.__reversed__(self)
 *
 *****************************************************************************/

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __len__)
    .set_num_inputs(1)
    .add_argument("self", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __getitem__)
    .set_num_inputs(2)
    .add_argument("self", "any_view", "")
    .add_argument("key", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __setitem__)
    .set_num_inputs(3)
    .add_argument("self", "any_view", "")
    .add_argument("key", "any_view", "")
    .add_argument("value", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __fused_getitem__)
    .set_num_inputs(2)
    .add_argument("self", "any_view", "")
    .add_argument("key", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __fused_setitem__)
    .set_num_inputs(3)
    .add_argument("self", "any_view", "")
    .add_argument("key", "*args", "")
    .add_argument("value", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __delitem__)
    .set_num_inputs(2)
    .add_argument("self", "any_view", "")
    .add_argument("key", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __getslice__)
    .set_num_inputs(4)
    .add_argument("self", "any_view", "")
    .add_argument("b", "any_view", "")
    .add_argument("e", "any_view", "")
    .add_argument("step", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __setslice__)
    .set_num_inputs(4)
    .add_argument("self", "any_view", "")
    .add_argument("b", "any_view", "")
    .add_argument("e", "any_view", "")
    .add_argument("item", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __contains__)
    .set_num_inputs(2)
    .add_argument("self", "any_view", "")
    .add_argument("key", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __hash__)
    .set_num_inputs(1)
    .add_argument("self", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __reversed__)
    .set_num_inputs(1)
    .add_argument("self", "any_view", "");

// attribute
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __getattr__)
    .set_num_inputs(2)
    .add_argument("self", "any_view", "")
    .add_argument("attr", "bytes_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(object, __setattr__)
    .set_num_inputs(3)
    .add_argument("self", "any_view", "")
    .add_argument("attr", "bytes_view", "")
    .add_argument("value", "any_view", "");

/******************************************************************************
 * builtin object's member function
 * Function schema :
 *    RTValue unbound_function(self, *args);
 *****************************************************************************/
#define MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(Prefix, OpName, ArgNum) \
  MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(Prefix, OpName)                          \
      .set_num_inputs(ArgNum)                                                            \
      .add_argument("self", "any_view", "")                                              \
      .add_argument("args", "*args", "")

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, append, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, add, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, extend, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, clear, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, find, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, decode, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, encode, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, lower, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, upper, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, isdigit, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, isalpha, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, split, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, join, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, replace, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, match, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, reserve, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, capacity, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, bucket_count, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, to_list, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, tolist, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, is_contiguous, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, reshape, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, squeeze, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, contiguous, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, shape, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, dtype, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, dim, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, device, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, keys, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, values, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, items, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, update, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, get, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, startswith, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, endswith, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, rstrip, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, lstrip, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, strip, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, count, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, pop, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, remove, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, insert, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, index, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, call, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, format, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, difference, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, difference_update, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, discard, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, union, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, transpose, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, as_type, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, reverse, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, sort, 2);

// Generic Trie
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, prefix_search, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, prefix_search_all, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, save, 2);
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC_OBJ_PYARGS(object, load, 2);

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
