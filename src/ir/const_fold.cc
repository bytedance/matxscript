// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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

/*!
 * \file int_set.cc
 * \brief The integer set functions
 */
#include "const_fold.h"

#include <matxscript/ir/prim_var.h>

namespace matxscript {
namespace arith {

PrimExpr SymbolicLimits::pos_inf_ = PrimVar("pos_inf", DataType::Handle());
PrimExpr SymbolicLimits::neg_inf_ = PrimVar("neg_inf", DataType::Handle());

}  // namespace arith
}  // namespace matxscript
