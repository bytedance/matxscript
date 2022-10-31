// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Taken from https://github.com/apache/tvm/blob/v0.7/include/tvm/ir/adt.h
 * with fixes applied:
 * - add namespace matx::ir for fix conflict with tvm
 * - remove TypeData
 * - add ClassType
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
 * \file matx/ir/adt.h
 * \brief Algebraic data type definitions.
 *
 * We adopt tvm relay's ADT definition as a unified class
 * for decripting structured data.
 */
#pragma once

#include <string>

#include <matxscript/ir/base.h>
#include <matxscript/ir/prim_var.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

/*!
 * \brief ADT constructor.
 * Constructors compare by pointer equality.
 * \sa Constructor
 */
class ConstructorNode : public HLOExprNode {
 public:
  /*! \brief The name (only a hint) */
  StringRef name_hint;
  /*! \brief Input to the constructor. */
  runtime::Array<Type> inputs;
  /*! \brief The datatype the constructor will construct. */
  GlobalTypeVar belong_to;
  /*! \brief Index in the table of constructors (set when the type is registered). */
  mutable int32_t tag = -1;

  ConstructorNode() {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("inputs", &inputs);
    v->Visit("belong_to", &belong_to);
    v->Visit("tag", &tag);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const ConstructorNode* other, SEqualReducer equal) const {
    // Use namehint for now to be consistent with the legacy relay impl
    // TODO(tvm-team) revisit, need to check the type var.
    return equal(name_hint, other->name_hint) && equal(inputs, other->inputs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name_hint);
    hash_reduce(inputs);
  }

  static constexpr const char* _type_key = "ir.Constructor";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ConstructorNode, HLOExprNode);
};

/*!
 * \brief Managed reference to ConstructorNode
 * \sa ConstructorNode
 */
class Constructor : public HLOExpr {
 public:
  /*!
   * \brief Constructor
   * \param name_hint the name of the constructor.
   * \param inputs The input types.
   * \param belong_to The data type var the constructor will construct.
   */
  MATX_DLL Constructor(Type ret_type,
                       StringRef name_hint,
                       runtime::Array<Type> inputs,
                       GlobalTypeVar belong_to);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Constructor, HLOExpr, ConstructorNode);
};

/*! \brief ClassType container node */
class ClassTypeNode : public TypeNode {
 public:
  uint64_t py_type_id;
  int64_t tag = 0;
  Type base;
  GlobalTypeVar header;
  runtime::Array<StringRef> var_names;
  runtime::Array<Type> var_types;
  runtime::Array<StringRef> func_names;
  runtime::Array<StringRef> unbound_func_names;
  runtime::Array<FuncType> func_types;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("tag", &py_type_id);
    v->Visit("tag", &tag);
    v->Visit("header", &header);
    v->Visit("var_names", &var_names);
    v->Visit("var_types", &var_types);
    v->Visit("func_names", &func_names);
    v->Visit("unbound_func_names", &unbound_func_names);
    v->Visit("func_types", &func_types);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ClassTypeNode* other, SEqualReducer equal) const {
    // disable recursive comparison var and function types for avoid endless loops
    if (this == other) {
      return true;
    }
    return py_type_id == other->py_type_id && tag == other->tag &&
           equal.DefEqual(header, other->header) && equal(var_names, other->var_names) &&
           equal(unbound_func_names, other->unbound_func_names) &&
           equal(func_names, other->func_names) && equal(base, other->base);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    // disable recursive comparison var and function types for avoid endless loops
    hash_reduce.DefHash(header);
    hash_reduce(tag);
    hash_reduce(py_type_id);
    hash_reduce(var_names);
    hash_reduce(unbound_func_names);
    hash_reduce(func_names);
    hash_reduce(base);
  }

  Type GetItem(const StringRef& name) const;

  runtime::Array<StringRef> GetVarNamesLookupTable() const;
  runtime::Array<Type> GetVarTypesLookupTable() const;

  runtime::Unicode GetPythonTypeName() const override {
    return header->name_hint.operator runtime::String().decode();
  }

  void ClearMembers() {
    var_names.clear();
    var_types.clear();
    unbound_func_names.clear();
    func_names.clear();
    func_types.clear();
    base = Type(nullptr);
    header = GlobalTypeVar(nullptr);
  }

  static constexpr const char* _type_key = "ir.ClassType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ClassTypeNode, TypeNode);
};

/*!
 * \brief Stores all data for an User Class.
 *
 */
class ClassType : public Type {
 public:
  /**
   * \brief ClassType Constructor
   * @param header the name of ClassType.
   * @param base the type of parent.
   * @param var_names member var names
   * @param var_types member var types
   * @param func_names member function global names
   * @param unbound_func_names member function origin names
   * @param func_types member function types
   */
  MATX_DLL ClassType(uint64_t py_type_id,
                     GlobalTypeVar header,
                     Type base,
                     runtime::Array<StringRef> var_names,
                     runtime::Array<Type> var_types,
                     runtime::Array<StringRef> func_names,
                     runtime::Array<StringRef> unbound_func_names,
                     runtime::Array<FuncType> func_types);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ClassType, Type, ClassTypeNode);
};

bool IsBaseTypeOf(const Type& base, const Type& derived, bool allow_same);

const PrimVar& GetImplicitClassSessionVar();

}  // namespace ir
}  // namespace matxscript
