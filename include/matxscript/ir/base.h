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

#include <matxscript/ir/span.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class BaseExprNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;
  /*!
   * \brief Stores the result of type inference(type checking).
   *
   * \note This can be undefined before type inference.
   *       This value is discarded during serialization.
   */
  mutable Type checked_type_ = Type(nullptr);
  /*!
   * \return The checked_type
   */
  const Type& checked_type() const;

  /*!
   * \brief Check if the inferred(checked) type of the Expr
   *  is backed by a TTypeNode and return it.
   *
   * \note This function will thrown an error if the node type
   *       of this Expr is not TTypeNode.
   *
   * \return The corresponding TTypeNode pointer.
   * \tparam The specific TypeNode we look for.
   */
  template <typename TTypeNode>
  inline const TTypeNode* type_as() const;

  static constexpr const char* _type_key = "BaseExpr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 58;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(BaseExprNode, Object);
};

/*!
 * \brief Managed reference to BaseExprNode.
 * \sa BaseExprNode
 */
class BaseExpr : public ObjectRef {
 public:
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(BaseExpr, ObjectRef, BaseExprNode);
};

/*!
 * \brief Base node of all primitive expressions.
 *
 *  A primitive expression deals with low-level
 *  POD data types and handles without
 *  doing life-cycle management for objects.
 *
 *  PrimExpr is used in the low-level code
 *  optimizations and integer analysis.
 *
 * \sa PrimExpr
 */
class PrimExprNode : public BaseExprNode {
 public:
  /*!
   * \brief The runtime data type of the primitive expression.
   *
   * runtime::DataType(dtype) provides coarse grained type information
   * during compile time and runtime. It is eagerly built in
   * PrimExpr expression construction and can be used for
   * quick type checking.
   *
   * dtype is sufficient to decide the Type of the PrimExpr
   * when it corresponds to POD value types such as i32.
   *
   * When dtype is DataType::Handle(), the expression could corresponds to
   * a more fine-grained Type, and we can get the type by running lazy type inference.
   */
  runtime::DataType dtype;

  static constexpr const char* _type_key = "PrimExpr";
  static constexpr const uint32_t _type_child_slots = 34;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(PrimExprNode, BaseExprNode);
};

/*!
 * \brief Reference to PrimExprNode.
 * \sa PrimExprNode
 */
class PrimExpr : public BaseExpr {
 public:
  /*!
   * \brief construct from integer.
   * \param value The value to be constructed.
   */
  MATX_DLL PrimExpr(int32_t value);  // NOLINT(*)
  /*!
   * \brief construct from float.
   * \param value The value to be constructed.
   */
  MATX_DLL PrimExpr(float value);  // NOLINT(*)

  /*! \return the data type of this expression. */
  runtime::DataType dtype() const {
    return static_cast<const PrimExprNode*>(get())->dtype;
  }

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimExpr, BaseExpr, PrimExprNode);
};

/*!
 * \brief Base node of all non-primitive expressions.
 *
 * HLOExpr supports tensor types, functions and ADT as
 * first class citizens. The life-cycle of the corresponding
 * objects are implicitly managed by the language.
 *
 * \sa HLOExpr
 */
class HLOExprNode : public BaseExprNode {
 public:
  static constexpr const char* _type_key = "HLOExpr";
  static constexpr const uint32_t _type_child_slots = 22;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(HLOExprNode, BaseExprNode);
};

/*!
 * \brief Managed reference to HLOExprNode.
 * \sa HLOExprNode
 */
class HLOExpr : public BaseExpr {
 public:
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOExpr, BaseExpr, HLOExprNode);
};

/*! \brief Base node of all statements. */
class StmtNode : public Object {
 public:
  Span span;

  static constexpr const char* _type_key = "Stmt";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 15;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(StmtNode, Object);
};

/*! \brief Container of all statements */
class Stmt : public ObjectRef {
 public:
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Stmt, ObjectRef, StmtNode);
};

template <typename TTypeNode>
inline const TTypeNode* BaseExprNode::type_as() const {
  static_assert(std::is_base_of<TypeNode, TTypeNode>::value,
                "TType must be a special case of type");
  MXCHECK(checked_type_.defined())
      << "Type inference for this Expr has not completed. Try to call infer_type pass.";
  const TTypeNode* node = checked_type_.as<TTypeNode>();
  MXCHECK(node != nullptr) << "Expected type to be " << TTypeNode::_type_key << ", but get "
                           << checked_type_->GetTypeKey();
  return node;
}

inline bool IsPrimType(const BaseExpr& t) {
  return IsPrimType(t->checked_type());
}

inline bool IsFloatType(const BaseExpr& t) {
  return IsFloatType(t->checked_type());
}

inline bool IsStringType(const BaseExpr& t) {
  return IsStringType(t->checked_type());
}

inline bool IsUnicodeType(const BaseExpr& t) {
  return IsUnicodeType(t->checked_type());
}

inline bool IsUnicodeRefType(const BaseExpr& t) {
  return IsRefType(t->checked_type()) &&
         IsUnicodeType(runtime::Downcast<RefType>(t->checked_type())->value);
}

inline bool IsListType(const BaseExpr& t) {
  return IsListType(t->checked_type());
}

inline bool IsDictType(const BaseExpr& t) {
  return IsDictType(t->checked_type());
}

inline bool IsSetType(const BaseExpr& t) {
  return IsSetType(t->checked_type());
}

inline bool IsObjectType(const BaseExpr& t) {
  return IsObjectType(t->checked_type());
}

inline bool IsIteratorType(const BaseExpr& t) {
  return IsIteratorType(t->checked_type());
}

inline bool IsFileType(const BaseExpr& t) {
  return IsFileType(t->checked_type());
}

}  // namespace ir
}  // namespace matxscript
