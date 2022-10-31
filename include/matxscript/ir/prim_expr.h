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
#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/base.h>
#include <matxscript/ir/prim_var.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/object_internal.h>

namespace matxscript {
namespace ir {

/*!
 * \brief Constant integer literals in the program.
 * \sa IntImm
 */
class IntImmNode : public PrimExprNode {
 public:
  /*! \brief the Internal value. */
  int64_t value;

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const IntImmNode* other, runtime::SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(runtime::SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "IntImm";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IntImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference class to IntImmNode.
 *
 * \sa IntImmNode
 */
class IntImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param dtype The data type of the value.
   * \param value The internal value.
   */
  MATX_DLL IntImm(runtime::DataType dtype, int64_t value, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(IntImm, PrimExpr, IntImmNode);
};

/*!
 * \brief Constant floating point literals in the program.
 * \sa FloatImm
 */
class FloatImmNode : public PrimExprNode {
 public:
  /*! \brief The constant value content. */
  double value;

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const FloatImmNode* other, runtime::SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(runtime::SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "FloatImm";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FloatImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference class to FloatImmNode.
 *
 * \sa FloatImmNode
 */
class FloatImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param dtype The data type of the value.
   * \param value The internal value.
   */
  MATX_DLL FloatImm(runtime::DataType dtype, double value, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(FloatImm, PrimExpr, FloatImmNode);
};

/*!
 * \brief Boolean constant.
 *
 *  This reference type is useful to add additional compile-time
 *  type checks and helper functions for Integer equal comparisons.
 */
class Bool : public IntImm {
 public:
  explicit Bool(bool value, Span span = Span()) : IntImm(runtime::DataType::Bool(), value) {
  }
  Bool operator!() const {
    return Bool((*this)->value == 0);
  }
  operator bool() const {
    return (*this)->value != 0;
  }

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Bool, IntImm, IntImmNode);
};

// Overload operators to make sure we have the most fine grained types.
inline Bool operator||(const Bool& a, bool b) {
  return Bool(a.operator bool() || b);
}
inline Bool operator||(bool a, const Bool& b) {
  return Bool(a || b.operator bool());
}
inline Bool operator||(const Bool& a, const Bool& b) {
  return Bool(a.operator bool() || b.operator bool());
}
inline Bool operator&&(const Bool& a, bool b) {
  return Bool(a.operator bool() && b);
}
inline Bool operator&&(bool a, const Bool& b) {
  return Bool(a && b.operator bool());
}
inline Bool operator&&(const Bool& a, const Bool& b) {
  return Bool(a.operator bool() && b.operator bool());
}

/*!
 * \brief Container of constant int that adds more constructors.
 *
 * This is used to store and automate type check
 * attributes that must be constant integer.
 *
 * \sa IntImm
 */
class Integer : public IntImm {
 public:
  Integer() {
  }
  /*!
   * \brief constructor from node.
   */
  explicit Integer(ObjectPtr<Object> node, Span span = Span()) : IntImm(node) {
  }
  /*!
   * \brief Construct integer from int value.
   */
  Integer(int value, Span span = Span()) : IntImm(runtime::DataType::Int(32), value) {
  }  // NOLINT(*)
  /*!
   * \brief Construct integer from int imm.
   * \param other The other value.
   */
  Integer(IntImm other, Span span = Span()) : IntImm(std::move(other)) {
  }  // NOLINT(*)
  /*!
   * \brief Constructor from enum
   * \tparam Enum The enum type.
   * \param value The enum value.
   */
  template <typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
  explicit Integer(Enum value, Span span = Span()) : Integer(static_cast<int>(value)) {
    static_assert(std::is_same<int, typename std::underlying_type<Enum>::type>::value,
                  "declare enum to be enum int to use visitor");
  }
  /*!
   * \brief Assign an expression to integer.
   * \param other another expression.
   */
  Integer& operator=(const IntImm& other) {
    data_ = runtime::ObjectInternal::GetObjectPtr(other);
    return *this;
  }
  /*!
   * \brief convert to int64_t
   */
  operator int64_t() const {
    MXCHECK(data_ != nullptr) << " Trying to reference a null Integer";
    return (*this)->value;
  }
  // comparators
  Bool operator==(int other) const {
    if (data_ == nullptr)
      return Bool(false);
    return Bool((*this)->value == other);
  }
  Bool operator!=(int other) const {
    return !(*this == other);
  }
  template <typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
  Bool operator==(Enum other) const {
    return *this == static_cast<int>(other);
  }
  template <typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
  Bool operator!=(Enum other) const {
    return *this != static_cast<int>(other);
  }
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class PrimCastNode : public PrimExprNode {
 public:
  /*! \brief Original data type. */
  PrimExpr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimCastNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.PrimCast";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimCastNode, PrimExprNode);
};

/*!
 * \brief Managed reference to PrimCastNode
 * \sa PrimCastNode
 */
class PrimCast : public PrimExpr {
 public:
  MATX_DLL PrimCast(runtime::DataType dtype, PrimExpr value, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimCast, PrimExpr, PrimCastNode);
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class HLOCastPrimNode : public PrimExprNode {
 public:
  /*! \brief Original data type. */
  BaseExpr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOCastPrimNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.HLOCastPrim";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOCastPrimNode, PrimExprNode);
};

/*!
 * \brief Managed reference to CastNode
 * \sa CastNode
 */
class HLOCastPrim : public PrimExpr {
 public:
  MATX_DLL HLOCastPrim(runtime::DataType type, BaseExpr value, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOCastPrim, PrimExpr, HLOCastPrimNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class PrimBinaryOpNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(T, PrimExprNode);
};

/*! \brief a + b */
class PrimAddNode : public PrimBinaryOpNode<PrimAddNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimAdd";
};

/*!
 * \brief Managed reference to AddNode
 * \sa AddNode
 */
class PrimAdd : public PrimExpr {
 public:
  MATX_DLL PrimAdd(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimAdd, PrimExpr, PrimAddNode);
};

/*! \brief a - b */
class PrimSubNode : public PrimBinaryOpNode<PrimSubNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimSub";
};

/*!
 * \brief Managed reference to SubNode
 * \sa SubNode
 */
class PrimSub : public PrimExpr {
 public:
  MATX_DLL PrimSub(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimSub, PrimExpr, PrimSubNode);
};

/*! \brief a * b */
class PrimMulNode : public PrimBinaryOpNode<PrimMulNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimMul";
};

/*!
 * \brief Managed reference to MulNode
 * \sa MulNode
 */
class PrimMul : public PrimExpr {
 public:
  MATX_DLL PrimMul(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimMul, PrimExpr, PrimMulNode);
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class PrimDivNode : public PrimBinaryOpNode<PrimDivNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimDiv";
};

/*!
 * \brief Managed reference to DivNode
 * \sa DivNode
 */
class PrimDiv : public PrimExpr {
 public:
  MATX_DLL PrimDiv(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimDiv, PrimExpr, PrimDivNode);
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class PrimModNode : public PrimBinaryOpNode<PrimModNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimMod";
};

/*!
 * \brief Managed reference to ModNode
 * \sa ModNode
 */
class PrimMod : public PrimExpr {
 public:
  MATX_DLL PrimMod(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimMod, PrimExpr, PrimModNode);
};

/*! \brief Floor division, floor(a/b) */
class PrimFloorDivNode : public PrimBinaryOpNode<PrimFloorDivNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimFloorDiv";
};

/*!
 * \brief Managed reference to FloorDivNode
 * \sa FloorDivNode
 */
class PrimFloorDiv : public PrimExpr {
 public:
  MATX_DLL PrimFloorDiv(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimFloorDiv, PrimExpr, PrimFloorDivNode);
};

/*! \brief The remainder of the floordiv */
class PrimFloorModNode : public PrimBinaryOpNode<PrimFloorModNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimFloorMod";
};

/*!
 * \brief Managed reference to FloorModNode
 * \sa FloorModNode
 */
class PrimFloorMod : public PrimExpr {
 public:
  MATX_DLL PrimFloorMod(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimFloorMod, PrimExpr, PrimFloorModNode);
};

/*! \brief min(a, b) */
class PrimMinNode : public PrimBinaryOpNode<PrimMinNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimMin";
};

/*!
 * \brief Managed reference to MinNode
 * \sa MinNode
 */
class PrimMin : public PrimExpr {
 public:
  MATX_DLL PrimMin(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimMin, PrimExpr, PrimMinNode);
};

/*! \brief max(a, b) */
class PrimMaxNode : public PrimBinaryOpNode<PrimMaxNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimMax";
};

/*!
 * \brief Managed reference to MaxNode
 * \sa MaxNode
 */
class PrimMax : public PrimExpr {
 public:
  MATX_DLL PrimMax(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimMax, PrimExpr, PrimMaxNode);
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class PrimCmpOpNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(T, PrimExprNode);
};

/*! \brief a == b */
class PrimEQNode : public PrimCmpOpNode<PrimEQNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimEQ";
};

/*!
 * \brief Managed reference to EQNode
 * \sa EQNode
 */
class PrimEQ : public PrimExpr {
 public:
  MATX_DLL PrimEQ(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimEQ, PrimExpr, PrimEQNode);
};

/*! \brief a != b */
class PrimNENode : public PrimCmpOpNode<PrimNENode> {
 public:
  static constexpr const char* _type_key = "ir.PrimNE";
};

/*!
 * \brief Managed reference to NENode
 * \sa NENode
 */
class PrimNE : public PrimExpr {
 public:
  MATX_DLL PrimNE(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimNE, PrimExpr, PrimNENode);
};

/*! \brief a < b */
class PrimLTNode : public PrimCmpOpNode<PrimLTNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimLT";
};

/*!
 * \brief Managed reference to LTNode
 * \sa LTNode
 */
class PrimLT : public PrimExpr {
 public:
  MATX_DLL PrimLT(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimLT, PrimExpr, PrimLTNode);
};

/*! \brief a <= b */
struct PrimLENode : public PrimCmpOpNode<PrimLENode> {
 public:
  static constexpr const char* _type_key = "ir.PrimLE";
};

/*!
 * \brief Managed reference to LENode
 * \sa LENode
 */
class PrimLE : public PrimExpr {
 public:
  MATX_DLL PrimLE(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimLE, PrimExpr, PrimLENode);
};

/*! \brief a > b */
class PrimGTNode : public PrimCmpOpNode<PrimGTNode> {
 public:
  static constexpr const char* _type_key = "ir.PrimGT";
};

/*!
 * \brief Managed reference to GTNode
 * \sa GTNode
 */
class PrimGT : public PrimExpr {
 public:
  MATX_DLL PrimGT(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimGT, PrimExpr, PrimGTNode);
};

/*! \brief a >= b */
class PrimGENode : public PrimCmpOpNode<PrimGENode> {
 public:
  static constexpr const char* _type_key = "ir.PrimGE";
};

/*!
 * \brief Managed reference to GENode
 * \sa GENode
 */
class PrimGE : public PrimExpr {
 public:
  MATX_DLL PrimGE(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimGE, PrimExpr, PrimGENode);
};

/*! \brief a && b */
class PrimAndNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimAndNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  static constexpr const char* _type_key = "ir.PrimAnd";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimAndNode, PrimExprNode);
};

/*!
 * \brief Managed reference to AndNode
 * \sa AndNode
 */
class PrimAnd : public PrimExpr {
 public:
  MATX_DLL PrimAnd(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimAnd, PrimExpr, PrimAndNode);
};

/*! \brief a || b */
class PrimOrNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimOrNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  static constexpr const char* _type_key = "ir.PrimOr";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimOrNode, PrimExprNode);
};

/*!
 * \brief Managed reference to OrNode
 * \sa OrNode
 */
class PrimOr : public PrimExpr {
 public:
  MATX_DLL PrimOr(PrimExpr a, PrimExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimOr, PrimExpr, PrimOrNode);
};

/*! \brief !a */
class PrimNotNode : public PrimExprNode {
 public:
  /*! \brief The input operand. */
  PrimExpr a;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("a", &a);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimNotNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(a, other->a);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
  }

  static constexpr const char* _type_key = "ir.PrimNot";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimNotNode, PrimExprNode);
};

/*!
 * \brief Managed reference to NotNode
 * \sa NotNode
 */
class PrimNot : public PrimExpr {
 public:
  MATX_DLL PrimNot(PrimExpr a, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimNot, PrimExpr, PrimNotNode);
};

/*!
 * \brief return true_value if condition is true, otherwise return false_value.
 * \note Both true_value and false_value could be evaluated
 *       regardless of the condition value.
 *       Do not use it to guard against out of bound access,
 *       please use if_then_else instead.
 */
class PrimSelectNode : public PrimExprNode {
 public:
  /*! \brief The condition */
  PrimExpr condition;
  /*! \brief value to be returned when condition is true. */
  PrimExpr true_value;
  /*! \brief value to be returned when condition is false. */
  PrimExpr false_value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("condition", &condition);
    v->Visit("true_value", &true_value);
    v->Visit("false_value", &false_value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimSelectNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(condition, other->condition) &&
           equal(true_value, other->true_value) && equal(false_value, other->false_value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(condition);
    hash_reduce(true_value);
    hash_reduce(false_value);
  }

  static constexpr const char* _type_key = "ir.PrimSelect";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimSelectNode, PrimExprNode);
};

/*!
 * \brief Managed reference to SelectNode
 * \sa SelectNode
 */
class PrimSelect : public PrimExpr {
 public:
  MATX_DLL PrimSelect(PrimExpr condition,
                      PrimExpr true_value,
                      PrimExpr false_value,
                      Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimSelect, PrimExpr, PrimSelectNode);
};

/*!
 * \brief Let binding. Bind var to value then evaluate body.
 */
class PrimLetNode : public PrimExprNode {
 public:
  /*! \brief The variable. */
  PrimVar var;
  /*! \brief The value to be binded. */
  PrimExpr value;
  /*! \brief The result expression. */
  PrimExpr body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimLetNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal.DefEqual(var, other->var) &&
           equal(value, other->value) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce.DefHash(var);
    hash_reduce(value);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "ir.PrimLet";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimLetNode, PrimExprNode);
};

/*!
 * \brief Managed reference to LetNode
 * \sa LetNode
 */
class PrimLet : public PrimExpr {
 public:
  MATX_DLL PrimLet(PrimVar var, PrimExpr value, PrimExpr body, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimLet, PrimExpr, PrimLetNode);
};

/*!
 * \brief Call node.
 */
class PrimCallNode : public PrimExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be matx::Op which corresponds to the primitive operators(intrinsics).
   *  - It can also be another function in the IRModule (GlobalVar).
   */
  HLOExpr op;

  /*! \brief The arguments. */
  runtime::Array<PrimExpr> args;
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimCallNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(op, other->op) && equal(args, other->args);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(op);
    hash_reduce(args);
  }

  static constexpr const char* _type_key = "ir.PrimCall";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimCallNode, PrimExprNode);
};

/*!
 * \brief Managed reference to CallNode
 * \sa CallNode
 */
class PrimCall : public PrimExpr {
 public:
  MATX_DLL PrimCall(runtime::DataType dtype,
                    HLOExpr op,
                    runtime::Array<PrimExpr> args,
                    Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimCall, PrimExpr, PrimCallNode);
};

}  // namespace ir
}  // namespace matxscript
