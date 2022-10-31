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

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/base.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

/*!
 * \brief The unique identifier of variables.
 *
 * Id is like name to the variables,
 * except that id is unique for each Var.
 *
 * \note Do not create Id directly, they are created in Var.
 */
class IdNode : public Object {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  StringRef name_hint;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
  }

  bool SEqualReduce(const IdNode* other, SEqualReducer equal) const {
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "Id";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IdNode, Object);
};

class Id : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param name_hint The name of the variable.
   */
  MATX_DLL explicit Id(StringRef name_hint);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Id, ObjectRef, IdNode);
};

/*!
 * \brief Local variables used in the let expression.
 *
 * Its semantics are similar to matx.Var node used in TVM's low level
 * tensor expression language.
 *
 * \note Each Var is bind only once and is immutable.
 */
/*! \brief Container for Var */
class HLOVarNode : public HLOExprNode {
 public:
  /*!
   * \brief The unique identifier of the Var.
   *
   * vid will be preserved for the same Var during type inference
   * and other rewritings, while the VarNode might be recreated
   * to attach additional information.
   * This property can be used to keep track of parameter Var
   * information across passes.
   */
  Id vid;
  /*!
   * \brief type annotation of the variable.
   * This field records user provided type annotation of the Var.
   * This field is optional and can be None.
   */
  Type type_annotation;

  /*! \return The name hint of the variable */
  const StringRef& name_hint() const {
    return vid->name_hint;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("vid", &vid);
    v->Visit("type_annotation", &type_annotation);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const HLOVarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(type_annotation, other->type_annotation) && equal(vid, other->vid);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(type_annotation);
    hash_reduce(vid);
  }

  static constexpr const char* _type_key = "ir.HLOVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 1;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(HLOVarNode, HLOExprNode);
};

class HLOVar : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param name_hint The name hint of a variable.
   * \param type_annotation The type annotation of a variable.
   * \param span The source span of the expression.
   */
  MATX_DLL HLOVar(StringRef name_hint, Type type_annotation, Span span = Span())
      : HLOVar(Id(name_hint), type_annotation, span) {
  }

  /*!
   * \brief The constructor
   * \param vid The unique id of a variable.
   * \param type_annotation The type annotation of a variable.
   * \param span The source span of the expression.
   */
  MATX_DLL HLOVar(Id vid, Type type_annotation, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOVar, HLOExpr, HLOVarNode);
};

class GlobalVar;
/*!
 * \brief Global variable that lives in the top-level module.
 *
 * A GlobalVar only refers to function definitions.
 * This is used to enable recursive calls between function.
 *
 * \sa GlobalVarNode
 */
class GlobalVarNode : public HLOExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  StringRef name_hint;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const GlobalVarNode* other, SEqualReducer equal) const {
    // name matters for global var.
    return equal(name_hint, other->name_hint) && equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name_hint);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "GlobalVar";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(GlobalVarNode, HLOExprNode);
};

/*!
 * \brief Managed reference to GlobalVarNode.
 * \sa GlobalVarNode
 */
class GlobalVar : public HLOExpr {
 public:
  MATX_DLL explicit GlobalVar(StringRef name_hint, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(GlobalVar, HLOExpr, GlobalVarNode);
};

}  // namespace ir
}  // namespace matxscript
