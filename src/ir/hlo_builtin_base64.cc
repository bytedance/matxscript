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

// base64
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(base64, b64encode)
    .set_num_inputs(2)
    .add_argument("s", "bytes_view", "")
    .add_argument("altchars", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(base64, b64decode)
    .set_num_inputs(3)
    .add_argument("s", "bytes_view", "")
    .add_argument("altchars", "any_view", "")
    .add_argument("validate", "int64", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
