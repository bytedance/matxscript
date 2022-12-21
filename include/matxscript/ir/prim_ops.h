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
 * \file matxscript/ir/op.h
 * \brief Common operators defined for Expr.
 *
 * \note Most of the operator defined here perform simple constant folding
 *   when the type is int32 or int64 for simplifying the index expressions.
 */
// Acknowledgement: Most operator APIs originate from Halide.
#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>

#include <matxscript/ir/op_expr.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/ir/stmt.h>
#include <matxscript/ir/type.h>

namespace matxscript {
namespace ir {

// Most common operators can be overloaded by argument type(PrimExpr).
// So we put them under the root namespace.
// It is also necessary to overload operators for PrimExpr.
//
// We put more developer oriented APIs -- make_const and is_const under tir
// as they are more specific to the tir namespace.

/*!
 * \brief Get the type of the expression under the unified type system.
 *
 * This function could return a more refined type than
 * the runtime type provided by expr->dtype
 *
 * \param expr The input parameter.
 * \return The result type.
 *
 * \sa matxscript/ir/type.h for discussion about the relation between Type and runtime::DataType.
 */
MATX_DLL Type GetType(const PrimExpr& expr);

/*!
 * \brief Get the implied DataType for storing values with type during runtime.
 *
 * \param type The input type.
 * \return The result runtime::DataType.
 *
 * \sa matxscript/ir/type.h for discussion about the relation between Type and runtime::DataType.
 */
MATX_DLL runtime::DataType GetRuntimeDataType(const Type& type);

/*!
 * Query the maximum possible value of dtype.
 * \param dtype The data type.
 * \return the maximum possible value in this format.
 */
MATX_DLL PrimExpr max_value(const runtime::DataType& dtype, Span span = Span());

/*!
 * Query the minimum possible value of dtype.
 * \param dtype The data type.
 * \return the minimum possible value in this format.
 */
MATX_DLL PrimExpr min_value(const runtime::DataType& dtype, Span span = Span());

/*!
 * Get the value of infinity.
 * \param dtype The data type.
 * \return the infinity value in this format.
 */
MATX_DLL PrimExpr infinity(const runtime::DataType& dtype, Span span = Span());

/*!
 * \brief cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
MATX_DLL PrimExpr cast(const runtime::DataType& t, PrimExpr value, Span span = Span());
/*!
 * \brief perform reinterpret cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
MATX_DLL PrimExpr reinterpret(const runtime::DataType& t, PrimExpr value, Span span = Span());
/*!
 * \brief add operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr add(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief subtraction operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr sub(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief negation.
 *
 * \param a input.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr neg(PrimExpr a, Span span = Span());
/*!
 * \brief multiplication operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr mul(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief division operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr div(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief left shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr left_shift(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief right shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr right_shift(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief greater
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr greater_than(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief greater_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr greater_or_equal(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief less
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr less_than(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief less_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr less_or_equal(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr equal(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief not_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr not_equal(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief and
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
MATX_DLL PrimExpr logic_and(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief or
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
MATX_DLL PrimExpr logic_or(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief not
 *
 * \param a left operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
MATX_DLL PrimExpr logic_not(PrimExpr a, Span span = Span());
/*!
 * \brief compute trunc(a / b)
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr truncdiv(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief compute the remainder of truncdiv
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr truncmod(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief compute floor(a / b) where a and b are non-negative.
 *
 * Use this function for index split calculation.
 *
 * This function might take advantage of the fact
 * that a and b are non-negative.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr indexdiv(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief compute the remainder floor(a / b) where a and b are non-negative.
 *
 * Use this function for index split calculation.
 * This function might take advantage of the fact
 * that a and b are non-negative.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr indexmod(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief a // b
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr floordiv(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief compute the remainder of floordiv
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr floormod(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief take maximum of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr max(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief take minimum of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr min(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief take bitwise and of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr bitwise_and(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief take bitwise or of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr bitwise_or(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief take bitwise xor of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, Span span = Span());
/*!
 * \brief take bitwise negation of two values
 *
 * \param a the input expression.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr bitwise_invert(PrimExpr a, Span span = Span());
/*!
 * \brief Conditional expression.
 *
 * \param cond The condition
 * \param true_value The value when results are true.
 * \param false_value The value when results are false.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
MATX_DLL PrimExpr if_then_else(PrimExpr cond,
                               PrimExpr true_value,
                               PrimExpr false_value,
                               Span span = Span());
/*!
 * \brief Mark condition as likely.
 * \param cond The condition
 * \return The marked expression.
 */
MATX_DLL PrimExpr likely(PrimExpr cond, Span span = Span());
/*!
 * \brief Calculate power(x, y)
 * \param x The left operand.
 * \param y The right operand.
 */
MATX_DLL PrimExpr pow(PrimExpr x, PrimExpr y, Span span = Span());
/*!
 * \brief Calculate absolute value of x.
 * \param x The input data
 *
 * \return The aboslute value of input data x
 */
MATX_DLL PrimExpr abs(PrimExpr x, Span span = Span());
/*!
 * \brief Check if x is NaN.
 * \param x The input data
 * \return The result expression.
 */
MATX_DLL PrimExpr isnan(PrimExpr x, Span span = Span());

/*!
 * \brief Check if x is finite.
 * \param x The input data
 * \return The result expression.
 */
MATX_DLL PrimExpr isfinite(PrimExpr x, Span span = Span());

/*!
 * \brief Check if x is infinite.
 * \param x The input data
 * \return The result expression.
 */
MATX_DLL PrimExpr isinf(PrimExpr x, Span span = Span());

/*!
 * \brief Calculate floor(x)
 * \param x The input expression.
 * \return The result expression.
 */
MATX_DLL PrimExpr floor(PrimExpr x, Span span = Span());

/*!
 * \brief Calculate ceil(x)
 * \param x The input expression.
 * \return The result expression.
 */
MATX_DLL PrimExpr ceil(PrimExpr x, Span span = Span());

/*!
 * \brief Calculate round(x)
 * \param x The input expression.
 * \return The result expression.
 */
MATX_DLL PrimExpr round(PrimExpr x, Span span = Span());

/*!
 * \brief Calculates std::nearbyint(x)
 * \param x The input expression.
 * \return The result expression.
 * This is a faster alternate to round.
 */
MATX_DLL PrimExpr nearbyint(PrimExpr x, Span span = Span());

/*!
 * \brief Calculate trunc(x)
 * \param x The input expression.
 * \return The result expression.
 */
MATX_DLL PrimExpr trunc(PrimExpr x, Span span = Span());

/*!
 * \brief Construct a large uint constant by its low 32 bits and high 32bits.
 * \param dtype The final data type.
 * \param low The lower 32 bits.
 * \param high The higher 32 bits.
 * \return The constructed expression.
 */
MATX_DLL PrimExpr LargeUIntImm(runtime::DataType dtype,
                               int64_t low,
                               int64_t high,
                               Span span = Span());

/*!
 * \brief Execute a multiplication between two Q-numbers x and y
 * followed by a right shift s. The mathematical expression is:
 *
 *    out = round(x*y*2^-s)
 *
 * Please note that the two Q-numbers x and y are supposed to have
 * the same number of fractional bits q.
 *
 * More about Q-numbers here: https://en.wikipedia.org/wiki/Q_(number_format)
 *
 * The rounding rule is to the nearest value, rounding half up
 * (i.e., round(x.1) = x and round (x.5) = x+1)
 * \param x first Q-number
 * \param y second Q-number
 * \param q number of fractional bits in x and y. Needs to be > 0
 * \param s integer right shift
 * \return The constructed expression.
 */
MATX_DLL PrimExpr
q_multiply_shift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr s, Span span = Span());

// Intrinsic operators
#define MATXSCRIPT_DECLARE_INTRIN_UNARY(OpName)                  \
  inline PrimExpr OpName(PrimExpr x, Span span) {                \
    static const Op& op = Op::Get("ir." #OpName);                \
    return ::matxscript::ir::PrimCall(x.dtype(), op, {x}, span); \
  }

MATXSCRIPT_DECLARE_INTRIN_UNARY(pow);
MATXSCRIPT_DECLARE_INTRIN_UNARY(exp);
MATXSCRIPT_DECLARE_INTRIN_UNARY(exp2);
MATXSCRIPT_DECLARE_INTRIN_UNARY(exp10);
MATXSCRIPT_DECLARE_INTRIN_UNARY(erf);
MATXSCRIPT_DECLARE_INTRIN_UNARY(tanh);
MATXSCRIPT_DECLARE_INTRIN_UNARY(sigmoid);
MATXSCRIPT_DECLARE_INTRIN_UNARY(sqrt);
MATXSCRIPT_DECLARE_INTRIN_UNARY(rsqrt);
MATXSCRIPT_DECLARE_INTRIN_UNARY(log);
MATXSCRIPT_DECLARE_INTRIN_UNARY(log2);
MATXSCRIPT_DECLARE_INTRIN_UNARY(log10);
MATXSCRIPT_DECLARE_INTRIN_UNARY(popcount);
MATXSCRIPT_DECLARE_INTRIN_UNARY(tan);
MATXSCRIPT_DECLARE_INTRIN_UNARY(cos);
MATXSCRIPT_DECLARE_INTRIN_UNARY(cosh);
MATXSCRIPT_DECLARE_INTRIN_UNARY(sin);
MATXSCRIPT_DECLARE_INTRIN_UNARY(sinh);
MATXSCRIPT_DECLARE_INTRIN_UNARY(asin);
MATXSCRIPT_DECLARE_INTRIN_UNARY(acos);
MATXSCRIPT_DECLARE_INTRIN_UNARY(atan);
MATXSCRIPT_DECLARE_INTRIN_UNARY(acosh);
MATXSCRIPT_DECLARE_INTRIN_UNARY(asinh);
MATXSCRIPT_DECLARE_INTRIN_UNARY(atanh);

#define MATXSCRIPT_DECLARE_INTRIN_BINARY(OpName)                    \
  inline PrimExpr OpName(PrimExpr x, PrimExpr y, Span span) {       \
    static const Op& op = Op::Get("ir." #OpName);                   \
    return ::matxscript::ir::PrimCall(x.dtype(), op, {x, y}, span); \
  }

MATXSCRIPT_DECLARE_INTRIN_BINARY(atan2);
MATXSCRIPT_DECLARE_INTRIN_BINARY(nextafter);
MATXSCRIPT_DECLARE_INTRIN_BINARY(copysign);
MATXSCRIPT_DECLARE_INTRIN_BINARY(hypot);
MATXSCRIPT_DECLARE_INTRIN_BINARY(ldexp);

/*!
 * \brief Check if type is a pointer to a runtime element type.
 * \param type The type to be checked.
 * \param element_type The corresponding element type.
 * \return The check results
 */
inline bool IsPointerType(const Type& type, const runtime::DataType& element_type) {
  if (!type.defined())
    return false;
  if (const auto* ptr_type = type.as<PointerTypeNode>()) {
    if (const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>()) {
      return prim_type->dtype == element_type;
    }
  }
  return false;
}

/*!
 * \brief Make a const value with certain data type.
 * \param t The target type.
 * \param value The input value
 * \return the result expression.
 * \tparam ValueType The constant value type
 */
template <typename ValueType,
          typename = typename std::enable_if<std::is_pod<ValueType>::value>::type>
inline PrimExpr make_const(runtime::DataType t, ValueType value, Span span = Span());
/*!
 * \brief Make a const zero expr.
 * \param t The target type.
 * \return the result expression.
 */
inline PrimExpr make_zero(runtime::DataType t, Span span = Span());
/*!
 * \brief Make a constant true expression.
 * \param lanes The number of lanes in the bool
 * \return The result expression.
 */
inline PrimExpr const_true(int lanes = 1, Span span = Span()) {
  return make_const(runtime::DataType::UInt(1, lanes), 1, span);
}
/*!
 * \brief Make a constant false expression.
 * \param lanes The number of lanes in the bool
 * \return The result expression.
 */
inline PrimExpr const_false(int lanes = 1, Span span = Span()) {
  return make_const(runtime::DataType::UInt(1, lanes), 0, span);
}
/*!
 * \brief Get x as constant int expression.
 * \param x The expression
 * \return the address to the int expression,
 *         return nullptr, if x is not IntImm.
 */
inline const int64_t* as_const_int(const PrimExpr& x) {
  if (!x.defined())
    return nullptr;
  if (const IntImmNode* op = x.as<IntImmNode>()) {
    return &(op->value);
  } else {
    return nullptr;
  }
}

/*!
 * \brief Check whether x is a constant integer expression.
 * \param x The input argument
 * \param value the value to be compared against.
 * \return whether x is constant expression.
 */
inline bool is_const_int(const PrimExpr& x, int64_t value);

/*!
 * \brief Check whether stmt is nop.
 * \param stmt The input statement
 * \return whether stmt is nop
 */
inline bool is_no_op(const Stmt& stmt);

/*!
 * \brief Check whether x is a constant integer 1
 * \param x The input argument.
 * \note This only return true for integer types.
 * \return whether x is constant 1
 */
inline bool is_one(const PrimExpr& x) {
  return is_const_int(x, 1);
}

/*!
 * \brief Check whether x is a constant integer 0
 * \param x The input argument
 * \return whether x is constant 0
 * \note This only return true for integer types.
 */
inline bool is_zero(const PrimExpr& x) {
  return is_const_int(x, 0);
}

/*!
 * \brief Check whether x is an integer constant.
 * \note This only return true for integer types.
 * \return whether x is constant
 */
inline bool is_const_int(const PrimExpr& x);

/*!
 * \brief Check whether x is an integer/float constant.
 * \note This only return true for integer types.
 * \return whether x is constant
 */
inline bool is_const_number(const PrimExpr& x);

/*!
 * \brief Left fold.
 * \param freduce The reduction function.
 * \param init_value The initial value.
 * \param values The values to be folded.
 * \return The result.
 * \tparam FReduce The type of the reduction.
 */
template <typename FReduce>
inline PrimExpr foldl(FReduce freduce, PrimExpr init_value, const runtime::Array<PrimExpr>& values);

/*!
 * \brief Check whether x is a constant power of two
 * If x is power of two, write the power to the shift.
 *
 * \param x The input expression.
 * \param shift The output shift if x is power of two.
 * \return whether x is constant power of two
 */
MATX_DLL bool is_const_power_of_two_integer(const PrimExpr& x, int* shift);

// Implementation details after this
inline bool is_const_int(const PrimExpr& x) {
  if (x.as<IntImmNode>()) {
    return true;
  }
  return false;
}

inline bool is_const_number(const PrimExpr& x) {
  if (x.as<IntImmNode>()) {
    return true;
  } else if (x.as<FloatImmNode>()) {
    return true;
  }
  return false;
}

inline bool is_positive_const(const PrimExpr& a) {
  if (const IntImmNode* op = a.as<IntImmNode>()) {
    return op->value > 0;
  } else {
    return false;
  }
}

inline bool is_negative_const(const PrimExpr& a) {
  if (const IntImmNode* op = a.as<IntImmNode>()) {
    return op->value < 0;
  } else {
    return false;
  }
}

inline bool is_const_int(const PrimExpr& x, int64_t value) {
  if (const auto* op = x.as<IntImmNode>()) {
    return op->value == value;
  }
  return false;
}

inline bool is_no_op(const Stmt& stmt) {
  if (!stmt.defined())
    return true;
  if (const auto* op = stmt.as<EvaluateNode>()) {
    return is_const_int(op->value);
  }
  if (const auto* op = stmt.as<SeqStmtNode>()) {
    return op->seq.size() == 0;
  }
  return false;
}

template <typename ValueType>
inline PrimExpr MakeConstScalar(runtime::DataType t, ValueType value, Span span = Span()) {
  if (t.is_int())
    return IntImm(t, static_cast<int64_t>(value), span);
  if (t.is_uint()) {
    // Use IntImm if it is a small integer
    uint64_t uval = static_cast<uint64_t>(value);
    if (uval <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return IntImm(t, static_cast<int64_t>(value), span);
    } else {
      uint64_t mask = (static_cast<uint64_t>(1) << 32U) - 1U;
      uint64_t low = uval & mask;
      uint64_t high = uval >> 32U;
      return LargeUIntImm(t, static_cast<int64_t>(low), static_cast<int64_t>(high), span);
    }
  }
  if (t.is_float() || t.is_bfloat16())
    return FloatImm(t, static_cast<double>(value), span);
  // For now, we store const scalar values of custom datatypes within doubles; later, during the
  // datatypes lowering pass, we will lower the value to its true representation in the format
  // specified by the datatype.
  // TODO(gus) when do we need to start worrying about doubles not being precise enough?
  if (static_cast<uint8_t>(t.code()) >= static_cast<uint8_t>(runtime::DataType::kCustomBegin)) {
    return FloatImm(t, static_cast<double>(value), span);
  }
  MXLOG(FATAL) << "cannot make const for type " << t;
  return PrimExpr();
}

template <typename ValueType, typename>
inline PrimExpr make_const(runtime::DataType t, ValueType value, Span span) {
  MXCHECK(t.lanes() == 1);
  return MakeConstScalar(t, value, span);
}

inline PrimExpr make_zero(runtime::DataType t, Span span) {
  if (t.is_handle()) {
    return reinterpret(t, make_const(runtime::DataType::UInt(64), 0, span), span);
  }
  return make_const(t, 0, span);
}

template <typename FReduce>
inline PrimExpr foldl(FReduce freduce,
                      PrimExpr init_value,
                      const runtime::Array<PrimExpr>& values) {
  for (PrimExpr val : values) {
    init_value = freduce(init_value, val);
  }
  return init_value;
}

// additional const expression overloading
#define MATXSCRIPT_DEFINE_ASSIGN_OP_OVERLOAD(Name, OpFunc)            \
  inline PrimExpr Name(PrimExpr& a, PrimExpr b, Span span = Span()) { \
    a = OpFunc(a, b, span);                                           \
    return a;                                                         \
  }

#define MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(Name)                                 \
  inline PrimExpr Name(const PrimExpr& a, float b, Span span = Span()) {                 \
    return Name(a, PrimExpr(b), span);                                                   \
  }                                                                                      \
  inline PrimExpr Name(float a, const PrimExpr& b, Span span = Span()) {                 \
    return Name(PrimExpr(a), b, span);                                                   \
  }                                                                                      \
  inline PrimExpr Name(int a, const PrimExpr& b, Span span = Span()) {                   \
    return Name(::matxscript::ir::make_const(b.dtype(), a), b, span);                    \
  }                                                                                      \
  inline PrimExpr Name(const PrimExpr& a, int b, Span span = Span()) {                   \
    return Name(a, ::matxscript::ir::make_const(a.dtype(), b), span);                    \
  }                                                                                      \
  inline PrimExpr Name(const PrimExpr& a, double b, Span span = Span()) {                \
    return Name(a, ::matxscript::ir::make_const(runtime::DataType::Float(64), b), span); \
  }

#define MATXSCRIPT_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(Name)           \
  inline PrimExpr Name(const PrimExpr& a, bool b, Span span = Span()) { \
    return Name(a, PrimExpr(b), span);                                  \
  }                                                                     \
  inline PrimExpr Name(bool a, const PrimExpr& b, Span span = Span()) { \
    return Name(PrimExpr(a), b, span);                                  \
  }

#define MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(Name)              \
  inline PrimExpr Name(const PrimExpr& a, int b, Span span = Span()) { \
    return Name(a, ::matxscript::ir::make_const(a.dtype(), b), span);  \
  }                                                                    \
  inline PrimExpr Name(int a, const PrimExpr& b, Span span = Span()) { \
    return Name(::matxscript::ir::make_const(b.dtype(), a), b, span);  \
  }

MATXSCRIPT_DEFINE_ASSIGN_OP_OVERLOAD(add_assign, add);
MATXSCRIPT_DEFINE_ASSIGN_OP_OVERLOAD(sub_assign, sub);
MATXSCRIPT_DEFINE_ASSIGN_OP_OVERLOAD(mul_assign, mul);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(add);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(sub);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(mul);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(max);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(min);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(div);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(greater_than);  // NOLINT(*)
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(greater_or_equal);
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(less_than);  // NOLINT(*)
MATXSCRIPT_DEFINE_BINOP_CONST_VAL_OVERLOAD(less_or_equal);
// integer related ops
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(indexdiv);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(indexmod);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(truncdiv);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(truncmod);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(floordiv);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(floormod);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(right_shift);  // NOLINT(*)
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(left_shift);   // NOLINT(*)
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(bitwise_and);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(bitwise_or);
MATXSCRIPT_DEFINE_INT_OP_CONST_VAL_OVERLOAD(bitwise_xor);
// logical ops
MATXSCRIPT_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(logic_and);
MATXSCRIPT_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(logic_or);

/*!
 * \brief Helper function to raise a compiler error about division ambiguity.
 * \note The call to this function will always results in a compiler error.
 * \tparam TA Any class type.
 */
template <typename TA>
inline void DivAmbiguityError(const TA& a) {
  constexpr bool div_ambiguity = !std::is_class<TA>::value;
  static_assert(div_ambiguity,
                "MATXScript supports multiple types of integer divisions, "
                "please call div, indexdiv/indexmod, "
                "floordiv/floormod or truncdiv/truncmod directly "
                "to avoid ambiguity in the code. "
                "Checkout these functions in expr_operator.h.");
}

// The following code are not intended to be used in the codebase.
// Instead, they generate clear compiler errors that ask developers
// to use the specific division function.
// The second template argument is necessary to make sure the
// code compiles lazily by the compiler during invocation.
template <typename TB>
inline PrimExpr operator/(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

template <typename TB>
inline PrimExpr operator/=(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

template <typename TB>
inline PrimExpr operator%(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

}  // namespace ir
}  // namespace matxscript
