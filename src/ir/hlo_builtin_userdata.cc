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

MATXSCRIPT_IR_DEFINE_HLO_METHOD(user_data, __getattr__, __getattr__)
    .set_num_inputs(2)
    .add_argument("self", "matx.NativeObject", "")
    .add_argument("attr", "str_view", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(user_data, __setattr__, set_attr)
    .set_num_inputs(3)
    .add_argument("self", "matx.NativeObject", "")
    .add_argument("attr", "str_view", "")
    .add_argument("val", "Any|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(user_data, call, generic_call)
    .set_num_inputs(2)
    .add_argument("self", "matx.NativeObject", "")
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_METHOD(user_data, call_attr, generic_call_attr)
    .set_num_inputs(3)
    .add_argument("self", "matx.NativeObject", "")
    .add_argument("func_name", "bytes_view", "")
    .add_argument("args", "*args", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
