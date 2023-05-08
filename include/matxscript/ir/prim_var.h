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
#include <matxscript/ir/range_expr.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

/*!
 * \brief A variable node in the IR.
 *
 * A variable is uniquely identified by its address.
 *
 * Each variable is only binded once in the following nodes:
 * - Allocate
 * - For
 * - Let
 * - LetStmt
 */
class PrimVarNode : public PrimExprNode {
 public:
  /*!
   * \brief The hint to the variable name.
   * \note Each variable is uniquely identified by its address.
   */
  StringRef name_hint;
  /*!
   * \brief type annotaion of the variable.
   *
   * It is an optional field that provides a refined type of the variable than dtype.
   *
   * \sa matxscript/ir/type.h for discussion of relations between runtime::DataType and Type.
   */
  Type type_annotation;

  void VisitAttrs(AttrVisitor* v) {
    PrimExprNode::VisitAttrs(v);
    v->Visit("name", &name_hint);
    v->Visit("type_annotation", &type_annotation);
  }

  bool SEqualReduce(const PrimVarNode* other, SEqualReducer equal) const {
    if (!PrimExprNode::SEqualReduce(other, equal))
      return false;
    if (!equal(type_annotation, other->type_annotation))
      return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    PrimExprNode::SHashReduce(hash_reduce);
    hash_reduce(type_annotation);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "ir.PrimVar";
  static constexpr const uint32_t _type_child_slots = 1;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(PrimVarNode, PrimExprNode);
};

/*! \brief a named variable in TIR */
class PrimVar : public PrimExpr {
 public:
  explicit PrimVar(ObjectPtr<Object> n) : PrimExpr(n) {
  }
  /*!
   * \brief Constructor
   * \param name_hint variable name
   * \param dtype data type
   */
  MATX_DLL explicit PrimVar(StringRef name_hint = "v",
                            runtime::DataType dtype = runtime::DataType::Int(32),
                            Span span = Span());
  /*!
   * \brief Constructor which provides a more detailed type annotation.
   * \param name_hint variable name.
   * \param type_annotation The type annotation.
   */
  MATX_DLL explicit PrimVar(StringRef name_hint, Type type_annotation, Span span = Span());

  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const PrimVarNode* operator->() const {
    return get();
  }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const PrimVarNode* get() const {
    return static_cast<const PrimVarNode*>(data_.get());
  }
  /*! \brief type indicate the container type */
  using ContainerType = PrimVarNode;
};

/******************************************************************************
 * Some Generic Structures consisting of PrimVar and PrimExpr
 *****************************************************************************/

/*!
 * \brief An iteration variable representing an iteration
 *  over a one dimensional interval.
 *
 *  The dtype of the extent of the `dom` of the IterVar must match the dtype of the internal Var.
 */
class PrimIterVarNode : public Object {
 public:
  /*!
   * \brief the domain of iteration, if known, can be None
   *  For the intermediate schedule node, before schedule.
   */
  RangeExpr dom;
  /*! \brief The looping variable */
  PrimVar var;
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dom", &dom);
    v->Visit("var", &var);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const PrimIterVarNode* other, SEqualReducer equal) const {
    return equal(dom, other->dom) && equal.DefEqual(var, other->var);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dom);
    hash_reduce.DefHash(var);
  }

  static constexpr const char* _type_key = "ir.PrimIterVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(PrimIterVarNode, Object);
};

/*!
 * \brief Iteration Variable,
 *  represents an iteration over an integer interval.
 *
 *  The dtype of the extent of the `dom` of the IterVar must match the dtype of the internal Var.
 */
class PrimIterVar : public ObjectRef {
 public:
  MATX_DLL PrimIterVar(RangeExpr dom, PrimVar var, Span span = Span());
  /*!
   * \return the corresponding var in the IterVar.
   */
  inline operator PrimExpr() const;

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimIterVar, ObjectRef, PrimIterVarNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(PrimIterVarNode);
};

}  // namespace ir
}  // namespace matxscript
