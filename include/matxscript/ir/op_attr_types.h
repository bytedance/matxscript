// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by Halide/TVM IR.
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
 * \file matx/ir/op_attr_types.h
 * \brief Attribute types in the Op registry for TIR ops.
 *
 * These attributes can be set via OpRegEntry::set_attr
 *
 * \sa matx/ir/op.h
 */
#pragma once

#include <matxscript/ir/prim_expr.h>
#include <matxscript/runtime/container.h>

namespace matxscript {
namespace ir {

/*!
 * \brief Global symbol of the op after lowering.
 */
using TGlobalSymbol = runtime::StringRef;
using TGlobalIsGenericBuiltinOp = bool;
using TGlobalIsGenericBuiltinSliceOp = bool;
using TGlobalIsExplicitContainerOp = bool;
using TKernelMethodName = runtime::StringRef;

/*!
 * \brief Whether the op is overloaded for vector form.
 */
using TVectorizable = bool;

/*!
 * \brief The effect type of the call.
 */
enum class CallEffectKind : int {
  /*! \brief Function corresponds to an annotation(e.g. likely) and can translate to identity. */
  kExprAnnotation = 0,
  /*!
   * \brief Pure function that do not interacts
   *        with any external state.
   */
  kPure = 1,
  /*!
   * \brief Function's that may read from states(e.g. RAM)
   */
  kReadState = 2,
  /*!
   * \brief Function that may read/write from states(e.g. RAM).
   */
  kUpdateState = 3,
  /*!
   * \brief Opaque function, cannot make any assumption
   */
  kOpaque = kUpdateState,
  /*!
   * \brief Special intrinsic to annotate call arguments info
   *        only valid as a direct argument to a call.
   */
  kSpecialCallArg = 4,
  /*!
   * \brief Embed opaque information in the Expr, cannot be codegen.
   */
  kEmbedInfo = 5
};

/*! \brief Use integer to record the kind. */
using TCallEffectKind = Integer;

}  // namespace ir
}  // namespace matxscript
