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
 * File builtin methods
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(file, open)
    .set_num_inputs(1)
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_HasNext, HasNext)
    .set_num_inputs(1)
    .add_argument("self", "matx.File", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_readline_string, ReadLineString)
    .set_num_inputs(1)
    .add_argument("self", "matx.File", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_readline_unicode, ReadLineUnicode)
    .set_num_inputs(1)
    .add_argument("self", "matx.File", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_readline, Next)
    .set_num_inputs(1)
    .add_argument("self", "matx.File", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_readlines, ReadLines)
    .set_num_inputs(1)
    .add_argument("self", "matx.File", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_read, Read)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "matx.File", "")
    .add_argument("size", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_read_bytes, ReadString)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "matx.File", "")
    .add_argument("size", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_read_unicode, ReadUnicode)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "matx.File", "")
    .add_argument("size", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(file_close, close)
    .set_num_inputs(1)
    .add_argument("self", "matx.File", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
