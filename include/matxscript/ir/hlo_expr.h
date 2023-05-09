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
#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/base.h>
#include <matxscript/ir/op_expr.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

class StringImmNode : public HLOExprNode {
 public:
  StringRef value;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const StringImmNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.StringImm";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(StringImmNode, HLOExprNode);
};

class StringImm : public HLOExpr {
 public:
  MATX_DLL StringImm(StringRef value, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(StringImm, HLOExpr, StringImmNode);
};

class UnicodeImmNode : public HLOExprNode {
 public:
  StringRef value;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const UnicodeImmNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.UnicodeImm";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(UnicodeImmNode, HLOExprNode);
};

class UnicodeImm : public HLOExpr {
 public:
  MATX_DLL UnicodeImm(StringRef value, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(UnicodeImm, HLOExpr, UnicodeImmNode);
};

/*!
 * \brief Represent a data type constant.
 */
class DataTypeImmNode : public HLOExprNode {
 public:
  /*! \brief The data value. */
  runtime::DataType value;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const DataTypeImmNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from data.
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.DataTypeImm";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DataTypeImmNode, HLOExprNode);
};

/*!
 * \brief Managed reference to DataTypeImm
 * \sa DataTypeImmNode
 */
class DataTypeImm : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param value The value input.
   * \param span The source span of the expression.
   */
  MATX_DLL explicit DataTypeImm(runtime::DataType value, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(DataTypeImm, HLOExpr, DataTypeImmNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(DataTypeImmNode);
};

/*! \brief A shape expression which allows users to construct a shape containing PrimExpr.
 */
class ShapeExprNode : public HLOExprNode {
 public:
  /*! The values of the shape expression. */
  Array<PrimExpr> values;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("values", &values);
  }

  bool SEqualReduce(const ShapeExprNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from values.
    return HLOExprNode::SEqualReduce(other, equal) && equal(values, other->values);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(values);
  }

  static constexpr const char* _type_key = "ir.ShapeExpr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ShapeExprNode, HLOExprNode);
};

class ShapeExpr : public HLOExpr {
 public:
  MATX_DLL explicit ShapeExpr(Array<PrimExpr> values, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ShapeExpr, HLOExpr, ShapeExprNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(ShapeExprNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class HLOBinaryOpNode : public HLOExprNode {
 public:
  /*! \brief The left operand. */
  BaseExpr a;
  /*! \brief The right operand. */
  BaseExpr b;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(a);
    hash_reduce(b);
  }

  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(T, HLOExprNode);
};

/*! \brief a + b */
class HLOAddNode : public HLOBinaryOpNode<HLOAddNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOAdd";
};

/*!
 * \brief Managed reference to AddNode
 * \sa AddNode
 */
class HLOAdd : public HLOExpr {
 public:
  MATX_DLL HLOAdd(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOAdd, HLOExpr, HLOAddNode);
};

/*! \brief a - b */
class HLOSubNode : public HLOBinaryOpNode<HLOSubNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOSub";
};

/*!
 * \brief Managed reference to SubNode
 * \sa SubNode
 */
class HLOSub : public HLOExpr {
 public:
  MATX_DLL HLOSub(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOSub, HLOExpr, HLOSubNode);
};

/*! \brief a * b */
class HLOMulNode : public HLOBinaryOpNode<HLOMulNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOMul";
};

/*!
 * \brief Managed reference to MulNode
 * \sa MulNode
 */
class HLOMul : public HLOExpr {
 public:
  MATX_DLL HLOMul(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOMul, HLOExpr, HLOMulNode);
};

/*! \brief Floor division, floor(a/b) */
class HLOFloorDivNode : public HLOBinaryOpNode<HLOFloorDivNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOFloorDiv";
};

/*!
 * \brief Managed reference to FloorDivNode
 * \sa FloorDivNode
 */
class HLOFloorDiv : public HLOExpr {
 public:
  MATX_DLL HLOFloorDiv(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOFloorDiv, HLOExpr, HLOFloorDivNode);
};

/*! \brief The remainder of the floordiv */
class HLOFloorModNode : public HLOBinaryOpNode<HLOFloorModNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOFloorMod";
};

/*!
 * \brief Managed reference to FloorModNode
 * \sa FloorModNode
 */
class HLOFloorMod : public HLOExpr {
 public:
  MATX_DLL HLOFloorMod(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOFloorMod, HLOExpr, HLOFloorModNode);
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class HLOCmpOpNode : public HLOExprNode {
 public:
  /*! \brief The left operand. */
  BaseExpr a;
  /*! \brief The right operand. */
  BaseExpr b;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(a);
    hash_reduce(b);
  }

  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(T, HLOExprNode);
};

/*! \brief a == b */
class HLOEqualNode : public HLOCmpOpNode<HLOEqualNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOEqual";
};

/*!
 * \brief Managed reference to HLOEqualNode
 * \sa HLOEqualNode
 */
class HLOEqual : public HLOExpr {
 public:
  MATX_DLL HLOEqual(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOEqual, HLOExpr, HLOEqualNode);
};

/*! \brief a != b */
class HLONotEqualNode : public HLOCmpOpNode<HLONotEqualNode> {
 public:
  static constexpr const char* _type_key = "ir.HLONotEqual";
};

/*!
 * \brief Managed reference to HLONotEqualNode
 * \sa HLONotEqualNode
 */
class HLONotEqual : public HLOExpr {
 public:
  MATX_DLL HLONotEqual(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLONotEqual, HLOExpr, HLONotEqualNode);
};

/*! \brief a < b */
class HLOLessThanNode : public HLOCmpOpNode<HLOLessThanNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOLessThan";
};

/*!
 * \brief Managed reference to HLOLessThanNode
 * \sa HLOLessThanNode
 */
class HLOLessThan : public HLOExpr {
 public:
  MATX_DLL HLOLessThan(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOLessThan, HLOExpr, HLOLessThanNode);
};

/*! \brief a <= b */
struct HLOLessEqualNode : public HLOCmpOpNode<HLOLessEqualNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOLessEqual";
};

/*!
 * \brief Managed reference to HLOLessEqualNode
 * \sa HLOLessEqualNode
 */
class HLOLessEqual : public HLOExpr {
 public:
  MATX_DLL HLOLessEqual(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOLessEqual, HLOExpr, HLOLessEqualNode);
};

/*! \brief a > b */
class HLOGreaterThanNode : public HLOCmpOpNode<HLOGreaterThanNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOGreaterThan";
};

/*!
 * \brief Managed reference to HLOGreaterThanNode
 * \sa HLOGreaterThanNode
 */
class HLOGreaterThan : public HLOExpr {
 public:
  MATX_DLL HLOGreaterThan(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOGreaterThan, HLOExpr, HLOGreaterThanNode);
};

/*! \brief a >= b */
class HLOGreaterEqualNode : public HLOCmpOpNode<HLOGreaterEqualNode> {
 public:
  static constexpr const char* _type_key = "ir.HLOGreaterEqual";
};

/*!
 * \brief Managed reference to HLOGreaterEqualNode
 * \sa HLOGreaterEqualNode
 */
class HLOGreaterEqual : public HLOExpr {
 public:
  MATX_DLL HLOGreaterEqual(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOGreaterEqual, HLOExpr, HLOGreaterEqualNode);
};

/*! \brief a && b */
class HLOAndNode : public HLOExprNode {
 public:
  /*! \brief The left operand. */
  BaseExpr a;
  /*! \brief The right operand. */
  BaseExpr b;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const HLOAndNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(a);
    hash_reduce(b);
  }

  static constexpr const char* _type_key = "ir.HLOAnd";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOAndNode, HLOExprNode);
};

/*!
 * \brief Managed reference to HLOAndNode
 * \sa HLOAndNode
 */
class HLOAnd : public HLOExpr {
 public:
  MATX_DLL HLOAnd(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOAnd, HLOExpr, HLOAndNode);
};

/*! \brief a || b */
class HLOOrNode : public HLOExprNode {
 public:
  /*! \brief The left operand. */
  BaseExpr a;
  /*! \brief The right operand. */
  BaseExpr b;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const HLOOrNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(a);
    hash_reduce(b);
  }

  static constexpr const char* _type_key = "ir.HLOOr";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOOrNode, HLOExprNode);
};

/*!
 * \brief Managed reference to HLOOrNode
 * \sa HLOOrNode
 */
class HLOOr : public HLOExpr {
 public:
  MATX_DLL HLOOr(BaseExpr a, BaseExpr b, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOOr, HLOExpr, HLOOrNode);
};

/*! \brief !a */
class HLONotNode : public HLOExprNode {
 public:
  /*! \brief The input operand. */
  BaseExpr a;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("a", &a);
  }

  bool SEqualReduce(const HLONotNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(a, other->a);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(a);
  }

  static constexpr const char* _type_key = "ir.HLONot";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLONotNode, HLOExprNode);
};

/*!
 * \brief Managed reference to HLONotNode
 * \sa HLONotNode
 */
class HLONot : public HLOExpr {
 public:
  MATX_DLL HLONot(BaseExpr a, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLONot, HLOExpr, HLONotNode);
};

/*!
 * \brief Call corresponds to operator invocation.
 *  Corresponds to the operator in computational graph terminology.
 */
class Call;
/*! \brief Call container. */
class CallNode : public HLOExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be matx::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  HLOExpr op;

  /*! \brief The arguments(inputs) of the call */
  Array<BaseExpr> args;

  /*!
   * \brief The type arguments passed to polymorphic(template) function.
   *
   * This is the advance feature that is only used when the function is
   * polymorphic. It is safe to be ignored in most cases. For example, in the
   * following code, the type_args of addone call is [int].
   *
   * \code
   *
   * template<typename T>
   * T addone(T a) { return a + 1; }
   *
   * void main() {
   *   int x = addone<int>(10);
   * }
   *
   * \endcode
   */
  Array<ObjectRef> type_args;  // type or IntImm

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("type_args", &type_args);
  }

  bool SEqualReduce(const CallNode* other, SEqualReducer equal) const {
    // skip type_args check for primitive ops.
    equal->MarkGraphNode();
    return HLOExprNode::SEqualReduce(other, equal) && equal(op, other->op) &&
           equal(args, other->args) && equal(type_args, other->type_args);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(op);
    hash_reduce(args);
    hash_reduce(type_args);
  }

  static constexpr const char* _type_key = "ir.Call";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(CallNode, HLOExprNode);
};

class Call : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param op The operator will be invoked.
   * \param args The arguments of the call.
   * \param attrs The attributes of the call node.
   * \param type_args The type arguments passed to a polymorphic function.
   * \param span The source span of the expression.
   */
  MATX_DLL Call(Type ret_type,
                HLOExpr op,
                Array<BaseExpr> args,
                Span span = Span(),
                Array<ObjectRef> type_args = Array<ObjectRef>());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Call, HLOExpr, CallNode);
};

/*! \brief HLOIterator container */
class HLOIteratorNode : public HLOExprNode {
 public:
  /*! \brief the iter of the HLOIterator */
  BaseExpr container;
  /*! \brief the iter of the HLOIterator */
  IntImm method;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("container", &container);
    v->Visit("method", &method);
  }

  bool SEqualReduce(const HLOIteratorNode* other, SEqualReducer equal) const {
    // specially handle empty HLOIterator as a constant is not a graph node.
    equal->MarkGraphNode();
    return HLOExprNode::SEqualReduce(other, equal) && equal(container, other->container);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(container);
  }

  static constexpr const char* _type_key = "ir.HLOIterator";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOIteratorNode, HLOExprNode);
};

class HLOIterator : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param iter The iterator of a HLOIterator.
   * \param span The source span of the expression.
   */
  MATX_DLL explicit HLOIterator(BaseExpr container, IntImm method, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOIterator, HLOExpr, HLOIteratorNode);
};

// Call args may be a InitializerList or InitializerDict
/*! \brief InitializerList container */
class InitializerListNode : public HLOExprNode {
 public:
  /*! \brief the fields of the InitializerList */
  Array<BaseExpr> fields;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("fields", &fields);
  }

  bool SEqualReduce(const InitializerListNode* other, SEqualReducer equal) const {
    if (!HLOExprNode::SEqualReduce(other, equal)) {
      return false;
    }
    // specially handle empty InitializerList as a constant is not a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    if (fields.size() != 0) {
      hash_reduce->MarkGraphNode();
      hash_reduce(fields);
    }
  }

  static constexpr const char* _type_key = "ir.InitializerList";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(InitializerListNode, HLOExprNode);
};

class InitializerList : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param fields The fields of a InitializerList.
   * \param span The source span of the expression.
   */
  MATX_DLL explicit InitializerList(Array<BaseExpr> fields, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(InitializerList, HLOExpr, InitializerListNode);
};

/*! \brief InitializerDict container */
class InitializerDictNode : public HLOExprNode {
 public:
  /*! \brief the fields of the InitializerDict */
  Map<BaseExpr, BaseExpr> fields;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("fields", &fields);
  }

  bool SEqualReduce(const InitializerDictNode* other, SEqualReducer equal) const {
    if (!HLOExprNode::SEqualReduce(other, equal)) {
      return false;
    }
    // specially handle empty InitializerDict as a constant is not a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    if (fields.size() != 0) {
      hash_reduce->MarkGraphNode();
      hash_reduce(fields);
    }
  }

  static constexpr const char* _type_key = "ir.InitializerDict";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(InitializerDictNode, HLOExprNode);
};

class InitializerDict : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param fields The fields of a InitializerDict.
   * \param span The source span of the expression.
   */
  MATX_DLL explicit InitializerDict(Map<BaseExpr, BaseExpr> fields, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(InitializerDict, HLOExpr, InitializerDictNode);
};

/*! \brief EnumAttr container */
class EnumAttrNode : public HLOExprNode {
 public:
  /*! \brief the val of the EnumAttr */
  StringRef enum_str;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("enum_str", &enum_str);
  }

  bool SEqualReduce(const EnumAttrNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && enum_str == other->enum_str;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(enum_str);
  }

  static constexpr const char* _type_key = "ir.EnumAttr";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(EnumAttrNode, HLOExprNode);
};

class EnumAttr : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param enum_str The value of a EnumAttr.
   * \param span The source span of the expression.
   */
  MATX_DLL explicit EnumAttr(StringRef enum_str, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(EnumAttr, HLOExpr, EnumAttrNode);
};

/*! \brief Get attr field out of a HLOExpr. */
class ClassGetItem;
class ClassGetItemNode : public HLOExprNode {
 public:
  /*! \brief The self Expression */
  HLOExpr self;
  /*! \brief which attr to get */
  StringImm attr;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("self", &self);
    v->Visit("attr", &attr);
  }

  bool SEqualReduce(const ClassGetItemNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(self, other->self) &&
           equal(attr, other->attr);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(self);
    hash_reduce(attr);
  }

  static constexpr const char* _type_key = "ir.ClassGetItem";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ClassGetItemNode, HLOExprNode);
};

class ClassGetItem : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param self The class to get an element from.
   * \param attr The attribute name for extracting a value in the tuple.
   * \param span The source span of the expression.
   */
  MATX_DLL ClassGetItem(HLOExpr self, StringImm attr, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ClassGetItem, HLOExpr, ClassGetItemNode);
};

/*!
 * \brief Cast value from one data type to another.
 */
class HLOCastNode : public HLOExprNode {
 public:
  /*! \brief Original data type. */
  BaseExpr value;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const HLOCastNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.HLOCast";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOCastNode, HLOExprNode);
};

/*!
 * \brief Managed reference to CastNode
 * \sa CastNode
 */
class HLOCast : public HLOExpr {
 public:
  MATX_DLL HLOCast(Type type, BaseExpr value, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOCast, HLOExpr, HLOCastNode);
};

/*!
 * \brief Move value from one to another.
 */
class HLOMoveNode : public HLOExprNode {
 public:
  /*! \brief Original data. */
  BaseExpr value;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const HLOMoveNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.HLOMove";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOMoveNode, HLOExprNode);
};

/*!
 * \brief Managed reference to HLOMoveNode
 * \sa HLOMoveNode
 */
class HLOMove : public HLOExpr {
 public:
  MATX_DLL HLOMove(BaseExpr value, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOMove, HLOExpr, HLOMoveNode);
};

/*!
 * \brief Enumerate one container.
 */
class HLOEnumerateNode : public HLOExprNode {
 public:
  /*! \brief Original container. */
  BaseExpr value;
  /*! \brief start. */
  PrimExpr start;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("start", &start);
  }

  bool SEqualReduce(const HLOEnumerateNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(value, other->value) &&
           equal(start, other->start);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(value);
    hash_reduce(start);
  }

  static constexpr const char* _type_key = "ir.HLOEnumerate";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOEnumerateNode, HLOExprNode);
};

/*!
 * \brief Managed reference to HLOEnumerateNode
 * \sa HLOEnumerateNode
 */
class HLOEnumerate : public HLOExpr {
 public:
  MATX_DLL HLOEnumerate(BaseExpr value, BaseExpr start, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOEnumerate, HLOExpr, HLOEnumerateNode);
};

/*!
 * \brief Zip multi containers.
 */
class HLOZipNode : public HLOExprNode {
 public:
  /*! \brief Original containers. */
  Array<BaseExpr> values;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("values", &values);
  }

  bool SEqualReduce(const HLOZipNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(values, other->values);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(values);
  }

  static constexpr const char* _type_key = "ir.HLOZip";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOZipNode, HLOExprNode);
};

/*!
 * \brief Managed reference to HLOZipNode
 * \sa HLOZipNode
 */
class HLOZip : public HLOExpr {
 public:
  MATX_DLL HLOZip(Array<BaseExpr> values, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOZip, HLOExpr, HLOZipNode);
};

/*!
 * \brief Representing the comprehension.
 */
class ComprehensionNode : public Object {
 public:
  BaseExpr target;
  BaseExpr iter;
  Array<BaseExpr> ifs;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("target", &target);
    v->Visit("iter", &iter);
    v->Visit("ifs", &ifs);
  }

  bool SEqualReduce(const ComprehensionNode* other, SEqualReducer equal) const {
    return equal(target, other->target) && equal(iter, other->iter) && equal(ifs, other->ifs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(target);
    hash_reduce(iter);
    hash_reduce(ifs);
  }

  static constexpr const char* _type_key = "ir.Comprehension";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ComprehensionNode, Object);
};

/*!
 * \brief Managed reference to ComprehensionNode.
 * \sa ComprehensionNode
 */
class Comprehension : public ObjectRef {
 public:
  MATX_DLL Comprehension(BaseExpr target, BaseExpr iter, Array<BaseExpr> ifs);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Comprehension, ObjectRef, ComprehensionNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(ComprehensionNode);
};

/*!
 * \brief Representing the ListComp.
 */
class ListCompNode : public HLOExprNode {
 public:
  BaseExpr elt;
  Array<Comprehension> generators;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("elt", &elt);
    v->Visit("generators", &generators);
  }

  bool SEqualReduce(const ListCompNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(elt, other->elt) &&
           equal(generators, other->generators);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(elt);
    hash_reduce(generators);
  }

  static constexpr const char* _type_key = "ir.ListComp";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ListCompNode, HLOExprNode);
};

/*!
 * \brief Managed reference to ListCompNode.
 * \sa ListCompNode
 */
class ListComp : public HLOExpr {
 public:
  MATX_DLL ListComp(Type ann_typed,
                    BaseExpr elt,
                    Array<Comprehension> generators,
                    Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ListComp, HLOExpr, ListCompNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(ListCompNode);
};

/*!
 * \brief Representing the SetComp.
 */
class SetCompNode : public HLOExprNode {
 public:
  BaseExpr elt;
  Array<Comprehension> generators;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("elt", &elt);
    v->Visit("generators", &generators);
  }

  bool SEqualReduce(const SetCompNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(elt, other->elt) &&
           equal(generators, other->generators);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(elt);
    hash_reduce(generators);
  }

  static constexpr const char* _type_key = "ir.SetComp";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SetCompNode, HLOExprNode);
};

/*!
 * \brief Managed reference to SetCompNode.
 * \sa SetCompNode
 */
class SetComp : public HLOExpr {
 public:
  MATX_DLL SetComp(Type ann_typed,
                   BaseExpr elt,
                   Array<Comprehension> generators,
                   Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(SetComp, HLOExpr, SetCompNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(SetCompNode);
};

/*!
 * \brief Representing the DictComp.
 */
class DictCompNode : public HLOExprNode {
 public:
  BaseExpr key;
  BaseExpr value;
  Array<Comprehension> generators;

  void VisitAttrs(AttrVisitor* v) {
    HLOExprNode::VisitAttrs(v);
    v->Visit("key", &key);
    v->Visit("value", &value);
    v->Visit("generators", &generators);
  }

  bool SEqualReduce(const DictCompNode* other, SEqualReducer equal) const {
    return HLOExprNode::SEqualReduce(other, equal) && equal(key, other->key) &&
           equal(value, other->value) && equal(generators, other->generators);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    HLOExprNode::SHashReduce(hash_reduce);
    hash_reduce(key);
    hash_reduce(value);
    hash_reduce(generators);
  }

  static constexpr const char* _type_key = "ir.DictComp";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DictCompNode, HLOExprNode);
};

/*!
 * \brief Managed reference to DictCompNode.
 * \sa DictCompNode
 */
class DictComp : public HLOExpr {
 public:
  MATX_DLL DictComp(Type ann_typed,
                    BaseExpr key,
                    BaseExpr value,
                    Array<Comprehension> generators,
                    Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(DictComp, HLOExpr, DictCompNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(DictCompNode);
};

}  // namespace ir
}  // namespace matxscript
