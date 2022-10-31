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

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const StringImmNode* other, runtime::SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(runtime::SHashReducer hash_reduce) const {
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
  ::matxscript::runtime::StringRef value;

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const UnicodeImmNode* other, runtime::SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(runtime::SHashReducer hash_reduce) const {
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.UnicodeImm";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(UnicodeImmNode, HLOExprNode);
};

class UnicodeImm : public HLOExpr {
 public:
  MATX_DLL UnicodeImm(runtime::StringRef value, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(UnicodeImm, HLOExpr, UnicodeImmNode);
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
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOAndNode* other, SEqualReducer equal) const {
    return equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
    v->Visit("a", &a);
    v->Visit("b", &b);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOOrNode* other, SEqualReducer equal) const {
    return equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
    v->Visit("a", &a);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLONotNode* other, SEqualReducer equal) const {
    return equal(a, other->a);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
  runtime::Array<BaseExpr> args;

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
  runtime::Array<ObjectRef> type_args;  // type or IntImm

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("type_args", &type_args);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const CallNode* other, SEqualReducer equal) const {
    // skip type_args check for primitive ops.
    equal->MarkGraphNode();
    return equal(op, other->op) && equal(args, other->args) && equal(type_args, other->type_args);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
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
                runtime::Array<BaseExpr> args,
                Span span = Span(),
                runtime::Array<ObjectRef> type_args = runtime::Array<ObjectRef>());

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
    v->Visit("container", &container);
    v->Visit("method", &method);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOIteratorNode* other, SEqualReducer equal) const {
    // specially handle empty HLOIterator as a constant is not a graph node.
    equal->MarkGraphNode();
    return equal(container, other->container);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
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
  runtime::Array<BaseExpr> fields;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const InitializerListNode* other, SEqualReducer equal) const {
    // specially handle empty InitializerList as a constant is not a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
  MATX_DLL explicit InitializerList(runtime::Array<BaseExpr> fields, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(InitializerList, HLOExpr, InitializerListNode);
};

/*! \brief InitializerDict container */
class InitializerDictNode : public HLOExprNode {
 public:
  /*! \brief the fields of the InitializerDict */
  runtime::Map<BaseExpr, BaseExpr> fields;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const InitializerDictNode* other, SEqualReducer equal) const {
    // specially handle empty InitializerDict as a constant is not a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
  MATX_DLL explicit InitializerDict(runtime::Map<BaseExpr, BaseExpr> fields, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(InitializerDict, HLOExpr, InitializerDictNode);
};

/*! \brief EnumAttr container */
class EnumAttrNode : public HLOExprNode {
 public:
  /*! \brief the val of the EnumAttr */
  StringRef enum_str;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("enum_str", &enum_str);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const EnumAttrNode* other, SEqualReducer equal) const {
    return enum_str == other->enum_str;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("self", &self);
    v->Visit("attr", &attr);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const ClassGetItemNode* other, SEqualReducer equal) const {
    return equal(self, other->self) && equal(attr, other->attr);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
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
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOCastNode* other, SEqualReducer equal) const {
    return equal(checked_type_, other->checked_type_) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(checked_type_);
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
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOMoveNode* other, SEqualReducer equal) const {
    return equal(checked_type_, other->checked_type_) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(checked_type_);
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
    v->Visit("value", &value);
    v->Visit("start", &start);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOEnumerateNode* other, SEqualReducer equal) const {
    return equal(checked_type_, other->checked_type_) && equal(value, other->value) &&
           equal(start, other->start);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(checked_type_);
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
  runtime::Array<BaseExpr> values;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("values", &values);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOZipNode* other, SEqualReducer equal) const {
    return equal(checked_type_, other->checked_type_) && equal(values, other->values);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(checked_type_);
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
  MATX_DLL HLOZip(runtime::Array<BaseExpr> values, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOZip, HLOExpr, HLOZipNode);
};

}  // namespace ir
}  // namespace matxscript
