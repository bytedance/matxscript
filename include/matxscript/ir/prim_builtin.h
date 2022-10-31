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
 * \file matx/ir/prim_builtin.h
 * \brief TIR builtin intrinsics.
 *
 * builtin intrinsics are stored as matx:Op.
 * They are processed in the same way as we process Ops.
 *
 * It is not necessary to create a function for every Op,
 * as we can obtain them through Op::Get.
 *
 * This file contains the most commonly used intrinsics or
 * those that have special semantics and need compiler support.
 */
#pragma once

#include <matxscript/ir/op_expr.h>

namespace matxscript {
namespace ir {

/*! \brief Collection of builtin intrinsics as ops */
namespace builtin {
/*!
 * \brief Reinterpret the value using the target type.
 */
MATX_DLL const Op& reinterpret();

/*!
 * \brief Marks a condition is likely going to happen.
 */
MATX_DLL const Op& likely();

/*!
 * \brief Bitwise and operator.
 */
MATX_DLL const Op& bitwise_and();

/*!
 * \brief Bitwise or operator.
 */
MATX_DLL const Op& bitwise_or();

/*!
 * \brief Bitwise xor operator.
 */
MATX_DLL const Op& bitwise_xor();

/*!
 * \brief Bitwise not operator.
 */
MATX_DLL const Op& bitwise_not();

/*!
 * \brief Left shift
 */
MATX_DLL const Op& shift_left();

/*!
 * \brief Right shift
 */
MATX_DLL const Op& shift_right();

/*!
 * \brief See pesudo code
 *
 *  Construct a big uint that may not be representable by int64
 *
 *  Expr large_uint_imm(uint32_t v0, uin32_t v1) {
 *    return (v1 << 32) | v0;
 *  }
 */
MATX_DLL const Op& large_uint_imm();

/*!
 * \brief Execute a multiplication between two Q-numbers x and y
 * followed by a right shift s
 * The default rounding rule is to the nearest value, rounding half up
 * (i.e., round(x.1) = x and round (x.5) = x+1)
 */
MATX_DLL const Op& q_multiply_shift();

/*!
 * \brief See pesudo code
 *
 *  Handle address_of(Load *op) {
 *     return &op->buffer_var[index];
 *  }
 */
MATX_DLL const Op& address_of();

/*!
 * \brief Same as select, used for unsafe memory access.
 *
 *  Type if_then_else(cond, a, b) {
 *    return cond ? a : b;
 *  }
 */
MATX_DLL const Op& if_then_else();

/*!
 * \brief See pesudo code
 *
 *  bool isnullptr(void* handle) {
 *     return handle == nullptr
 *  }
 */
MATX_DLL const Op& isnullptr();

/*!
 * \brief Check if value is nan
 */
MATX_DLL const Op& isnan();

/*!
 * \brief Popcount
 */
MATX_DLL const Op& popcount();

/*!
 * \brief Fused multiply add
 *
 *  Type fma(a, b, c) {
 *    return a * b + c;
 *  }
 */
MATX_DLL const Op& fma();

/*!
 * \brief Call an extern C function with given name
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_extern(name, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This intrinsic does not provide any type checking,
 *       and is main used for backward compatibility reasons.
 *       Always consider use pre-registered and typed matx::Op first.
 */
MATX_DLL const Op& call_extern();

/*!
 * \brief Call an pure extern C function with given name
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_pure_extern(name, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This intrinsic does not provide any type checking,
 *       and is main used for backward compatibility reasons.
 *       Always consider use pre-registered and typed matx::Op first.
 */
MATX_DLL const Op& call_pure_extern();

/*!
 * \brief Call an LLVM intrinsic with a given intrinsic id
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_llvm_pure_intrin(intrin_id, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This op does not provide any type checking.
 */
MATX_DLL const Op& call_llvm_intrin();

/*!
 * \brief Call an LLVM pure intrinsic with a given intrinsic id
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_llvm_pure_intrin(intrin_id, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This op does not provide any type checking.
 */
MATX_DLL const Op& call_llvm_pure_intrin();

/*!
 * \brief Call an SPIRV pure GLSL450 intrinsic.
 *
 *  Type call_spirv_pure_glsl450(intrin_id, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This op does not provide any type checking.
 */
MATX_DLL const Op& call_spirv_pure_glsl450();

// TODO(tvm-team) revisit the builtins below
// some of them can simply become ops with special codegen attr.
/*!
 * \brief Prefetch a cacheline
 */
MATX_DLL const Op& prefetch();

// TODO(tvm-team) replace the usage of the vector operations by Shuffle.
/*!
 * \brief Get the high level half of the vector
 */
MATX_DLL const Op& vectorhigh();

/*!
 * \brief Get the low-level half of the vector
 */
MATX_DLL const Op& vectorlow();

/*!
 * \brief Concat two vectors.
 */
MATX_DLL const Op& vectorcombine();

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
