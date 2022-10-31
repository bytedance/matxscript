// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Taken from https://github.com/apache/tvm/blob/v0.7/include/tvm/tir/analysis.h
 * with fixes applied:
 * - add namespace matx::ir for fix conflict with tvm
 * - remove unused code
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
 * \file matx/ir/analysis.h
 * \brief Analysis utilitie and passes for TIR.
 */
#pragma once

#include <string>

#include <matxscript/ir/expr.h>
#include <matxscript/ir/stmt.h>

namespace matxscript {
namespace ir {

/*!
 * \brief Compare two expressions recursively and check if they are equal
 *        to each other without var remapping.
 *
 *  This function does not remap variable bindings, it will not
 *  return true for (let x = 1 in x + 1) vs (let y = 1 in y + 1), unless x.same_as(y).
 *
 *  Use StructuralEqual for such cases.
 *
 *  Due to the restriction of not remapping variables, this function can run
 *  faster than StructuralEqual and can be used as a utility function during arithmetic
 *  simplifications.
 *
 * \sa StructuralEqual
 */
struct ExprDeepEqual {
 public:
  MATX_DLL bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const;
};

}  // namespace ir
}  // namespace matxscript
