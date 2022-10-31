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

// random
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, random).set_num_inputs(0);

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, seed)
    .set_num_inputs(1)
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, uniform)
    .set_num_inputs(2)
    .add_argument("a", "float", "")
    .add_argument("b", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, triangular)
    .set_num_inputs(1)
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, randint)
    .set_num_inputs(2)
    .add_argument("a", "int", "")
    .add_argument("b", "int", "");

// MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, choice).set_num_inputs(1);
// MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, randrange).set_num_inputs(0);
// MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, sample).set_num_inputs(-1);
// MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, shuffle).set_num_inputs(-1);
// MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, choices).set_num_inputs(-1);

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, normalvariate)
    .set_num_inputs(2)
    .add_argument("mu", "float", "")
    .add_argument("sigma", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, lognormvariate)
    .set_num_inputs(2)
    .add_argument("mu", "float", "")
    .add_argument("sigma", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, expovariate)
    .set_num_inputs(1)
    .add_argument("lambda", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, vonmisesvariate)
    .set_num_inputs(2)
    .add_argument("mu", "float", "")
    .add_argument("kappa", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, gammavariate)
    .set_num_inputs(2)
    .add_argument("alpha", "float", "")
    .add_argument("beta", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, gauss)
    .set_num_inputs(2)
    .add_argument("mu", "float", "")
    .add_argument("sigma", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, betavariate)
    .set_num_inputs(2)
    .add_argument("alpha", "float", "")
    .add_argument("beta", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, paretovariate)
    .set_num_inputs(1)
    .add_argument("alpha", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, weibullvariate)
    .set_num_inputs(2)
    .add_argument("alpha", "float", "")
    .add_argument("beta", "float", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, getstate).set_num_inputs(0);

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, setstate)
    .set_num_inputs(1)
    .add_argument("state", "Tuple", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(random, getrandbits)
    .set_num_inputs(1)
    .add_argument("k", "int", "");

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
