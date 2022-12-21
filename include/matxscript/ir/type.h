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
 * \file matx/ir/type.h
 * \brief IR/AST nodes for the unified type system in TVM.
 *
 * This file contains types that are common across IR variants.
 *
 * ## Relation between Type and runtime::DataType
 *
 * Besides Type, we also store a dtype field in the low-level PrimExpr.
 * runtime::DataType(dtype) provides coarse grained type information
 * during compile time and runtime. It is eagerly built in
 * low-level expression construction and can be used for
 * quick type checking in the low-level IR.
 * For example, when an Expr's dtype is int32,
 * we know for sure that its type is also int32.
 *
 * On the other hand, Type provides more fine grained information.
 * For example, a low level expression can have DataType::Handle() as
 * its dtype and MemRef[float32] as its type.
 * Types are usually lazily constructed via type checking,
 * so they may not readily be available during IR construction.
 *
 * The unified Type serves as a common bridge across IR dialects.
 * For example, we require all the functions to have a type signature,
 * which allow us to build cross dialect function calls.
 */

#pragma once

#include <string>

#include <matxscript/ir/_base/structural_equal.h>
#include <matxscript/ir/span.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

using runtime::AttrVisitor;
using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;
using runtime::SEqualReducer;
using runtime::SHashReducer;

/*!
 * \brief Type is the base type of all types.
 *
 * Relay's type system contains following subclasses:
 *
 * - PrimType: type of primitive type values used in the low-level IR.
 * - FuncType: type of a function.
 * - TensorType: type of certain Tensor values in the expression.
 *
 * There are also advanced types to support generic(polymorphic types).
 * \sa Type
 */
class TypeNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  /*!
   * \brief Whether the instance is an iterable object.
   *        This means that the HasNext and Next methods need to be implemented
   * @return
   */
  virtual bool Iterable() const {
    return false;
  }

  /*!
   * \brief Whether the instance has begin and end method.
   *
   * @return
   */
  virtual bool HasBeginEnd() const {
    return false;
  }

  virtual runtime::Unicode GetPythonTypeName() const {
    return U"Any";
  }

  virtual bool IsFullTyped() const {
    return false;
  }

  static constexpr const char* _type_key = "Type";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 14;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(TypeNode, Object);
};

/*!
 * \brief Managed reference to TypeNode.
 * \sa TypeNode
 */
class Type : public ObjectRef {
 public:
  bool operator==(const Type& other) const {
    return runtime::StructuralEqual()(*this, other);
  }
  bool operator!=(const Type& other) const {
    return !operator==(other);
  }
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Type, ObjectRef, TypeNode);
};

/*!
 * \brief Get the implied DataType for storing values with type during runtime.
 *
 * \param type The input type.
 * \return The result runtime::DataType.
 *
 * \sa matxscript/ir/type.h for discussion about the relation between Type and runtime::DataType.
 */
MATX_DLL runtime::DataType GetRuntimeDataType(const Type& type);
MATX_DLL bool IsRuntimeDataType(const Type& type);

/*!
 * \brief Primitive data types used in the low-level IR.
 *
 * PrimType represents POD-values and handles that are
 * not automatically managed by the runtime.
 *
 * \sa PrimType
 */
class PrimTypeNode : public TypeNode {
 public:
  /*!
   * \brief The corresponding dtype field.
   */
  runtime::DataType dtype;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
  }

  bool SEqualReduce(const PrimTypeNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
  }

  runtime::Unicode GetPythonTypeName() const override {
    if (dtype.is_bool()) {
      return U"bool";
    } else if (dtype.is_float()) {
      return U"float";
    } else if (dtype.is_int()) {
      return U"int";
    } else if (dtype.is_handle()) {
      return U"pointer";
    } else if (dtype.is_bfloat16()) {
      return U"bfloat16";
    }
    return U"CustomPrimType";
  }

  static constexpr const char* _type_key = "PrimType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PrimTypeNode.
 * \sa PrimTypeNode
 */
class PrimType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param dtype The corresponding dtype.
   */
  MATX_DLL explicit PrimType(runtime::DataType dtype);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimType, Type, PrimTypeNode);
};

inline Type BoolType() {
  return PrimType(runtime::DataType::Bool());
}

/*!
 * \brief Low-level raw pointer type.
 *
 *  PointerType represents type hints in the TIR to be
 *  passed to the final code generator.
 *
 *  PointerType should not occur in the high-level analysis.
 *
 * \sa PointerType
 */
class PointerTypeNode : public TypeNode {
 public:
  /*!
   * \brief The type of the element which the pointer points to.
   */
  Type element_type;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("element_type", &element_type);
  }

  bool SEqualReduce(const PointerTypeNode* other, SEqualReducer equal) const {
    return equal(element_type, other->element_type);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(element_type);
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"pointer";
  }

  static constexpr const char* _type_key = "PointerType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PointerTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PointerTypeNode.
 * \sa PointerTypeNode
 */
class PointerType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param element_type The type of the element which the pointer points to.
   */
  MATX_DLL explicit PointerType(Type element_type);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PointerType, Type, PointerTypeNode);
};

/*! \brief Possible kinds of TypeVars. */
enum TypeKind : int {
  kType = 0,
  /*! \brief Template variable in shape expression. */
  kShapeVar = 1,
  kBaseType = 2,
  kConstraint = 4,
  kAdtHandle = 5
};

/*!
 * \brief Type parameter in functions.
 *
 * A type variable can be viewed as template parameter in c++ template function.
 *
 * For example, in the following pesudo code,
 * the TypeVar of f is TypeVar("n", kind=kShapeVar).
 * This function can take in a Tensor with shape=(3, 3) and
 * returns a Tensor with shape=(9,)
 *
 * \code
 *
 *  template<i32 n>
 *  f(x : Tensor[i32, (n, n)]) -> Tensor[i32, (n * n)]
 *
 * \endcode
 * \sa TypeVar, TypeKind
 */
class TypeVarNode : public TypeNode {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  StringRef name_hint;
  /*! \brief The kind of type parameter */
  TypeKind kind;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TypeVarNode* other, SEqualReducer equal) const {
    return equal(kind, other->kind) && equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(kind);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "TypeVar";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TypeVarNode, TypeNode);
};

/*!
 * \brief Managed reference to TypeVarNode
 * \sa TypeVarNode
 */
class TypeVar : public Type {
 public:
  /*!
   * \brief Constructor
   * \param name_hint The name of the type var.
   * \param kind The kind of the type var.
   * \param span The span information.
   */
  MATX_DLL TypeVar(StringRef name_hint, TypeKind kind, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(TypeVar, Type, TypeVarNode);
};

/*!
 * \brief A global type variable that is used for defining new types or type aliases.
 * \sa GlobalTypeVar
 */
class GlobalTypeVarNode : public TypeNode {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  StringRef name_hint;
  /*! \brief The kind of type parameter */
  TypeKind kind;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("kind", &kind);
  }

  bool SEqualReduce(const GlobalTypeVarNode* other, SEqualReducer equal) const {
    // name matters for now in global type var.
    return equal(name_hint, other->name_hint) && equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name_hint);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "GlobalTypeVar";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(GlobalTypeVarNode, TypeNode);
};

/*!
 * \brief Managed reference to GlobalTypeVarNode
 * \sa GlobalTypeVarNode
 */
class GlobalTypeVar : public Type {
 public:
  /*!
   * \brief Constructor
   * \param name_hint The name of the type var.
   * \param kind The kind of the type var.
   * \param span The span of the type.
   */
  MATX_DLL GlobalTypeVar(StringRef name_hint, TypeKind kind, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(GlobalTypeVar, Type, GlobalTypeVarNode);
};

/*!
 * \brief The type of tuple values.
 * \sa TupleType
 */
class TupleTypeNode : public TypeNode {
 public:
  bool is_std_tuple = false;
  /*! \brief The type of each field in the tuple. */
  runtime::Array<Type> fields;

  TupleTypeNode() {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TupleTypeNode* other, SEqualReducer equal) const {
    return is_std_tuple == other->is_std_tuple && equal(fields, other->fields);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(is_std_tuple);
    hash_reduce(fields);
  }

  bool Iterable() const override {
    return true;
  }

  bool HasBeginEnd() const override {
    return true;
  }

  runtime::Unicode GetPythonTypeName() const override {
    if (fields.empty()) {
      return U"Tuple";
    }
    std::stringstream ss;
    ss << "Tuple[";
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << fields[i];
    }
    ss << "]";
    return runtime::String(ss.str()).decode();
  }

  static constexpr const char* _type_key = "TupleType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TupleTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TupleTypeNode.
 * \sa TupleTypeNode.
 */
class TupleType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param fields Fields in the tuple.
   * \param span The span of the type.
   */
  MATX_DLL explicit TupleType(runtime::Array<Type> fields, Span span = Span())
      : TupleType(std::move(fields), false, std::move(span)) {
  }

  MATX_DLL explicit TupleType(runtime::Array<Type> fields, bool is_std_tuple, Span span = Span());

  /*!
   * \brief Create an empty tuple type that constains nothing.
   * \return A empty tuple type.
   */
  MATX_DLL TupleType static Empty();

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(TupleType, Type, TupleTypeNode);
};

/*!
 * \return a type that represents void.
 */
inline Type VoidType() {
  return TupleType::Empty();
}

/*!
 * \brief Check whether the tyep represents void.
 * \return The check result.
 */
inline bool IsVoidType(const Type& type) {
  auto* n = type.as<TupleTypeNode>();
  return n && n->fields.size() == 0;
}

/*!
 * \brief Potential Constraints in a function.
 * \sa TypeConstraint
 */
class TypeConstraintNode : public TypeNode {
 public:
  static constexpr const char* _type_key = "TypeConstraint";
  static constexpr const uint32_t _type_child_slots = 1;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(TypeConstraintNode, TypeNode);
};

/*!
 * \brief Managed reference to TypeConstraintNode.
 * \sa TypeConstraintNode, TypeRelation
 */
class TypeConstraint : public Type {
 public:
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(TypeConstraint, Type, TypeConstraintNode);
};

/*!
 * \brief Function type.
 *
 * We support polymorphic function type.
 * This can be roughly viewed as template function in C++.
 *
 * \sa FuncType, TypeVar, TypeConstraint
 */
class FuncTypeNode : public TypeNode {
 public:
  /*! \brief type type of arguments */
  runtime::Array<Type> arg_types;
  /*! \brief The type of return value. */
  Type ret_type;
  // The following fields are used in polymorphic(template) functions
  // For normal functions, the following two fields will be empty.
  /*! \brief The type parameters of the function */
  runtime::Array<TypeVar> type_params;
  /*!
   * \brief potential constraint the type need to obey
   * \note this field is reserved for futher purposes.
   */
  runtime::Array<TypeConstraint> type_constraints;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("arg_types", &arg_types);
    v->Visit("ret_type", &ret_type);
    v->Visit("type_params", &type_params);
    v->Visit("type_constraints", &type_constraints);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const FuncTypeNode* other, SEqualReducer equal) const {
    // type params first as they defines type vars.
    return equal.DefEqual(type_params, other->type_params) && equal(arg_types, other->arg_types) &&
           equal(ret_type, other->ret_type) && equal(type_constraints, other->type_constraints);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(type_params);
    hash_reduce(arg_types);
    hash_reduce(ret_type);
    hash_reduce(type_constraints);
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"function";
  }

  static constexpr const char* _type_key = "FuncType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FuncTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to FuncTypeNode.
 * \sa FuncTypeNode
 */
class FuncType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param arg_types The types of the arguments.
   * \param ret_type The type of the return value.
   * \param type_params The type parameters.
   * \param type_constraints The type constraints.
   * \param span The span information.
   * \sa FuncTypeNode for more docs about these fields.
   */
  MATX_DLL FuncType(runtime::Array<Type> arg_types,
                    Type ret_type,
                    runtime::Array<TypeVar> type_params,
                    runtime::Array<TypeConstraint> type_constraints,
                    Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(FuncType, Type, FuncTypeNode);
};

class ObjectTypeNode : public TypeNode {
 public:
  bool is_view = false;

  ObjectTypeNode() {
  }

  bool Iterable() const override {
    return true;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("is_view", &is_view);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ObjectTypeNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  static constexpr const char* _type_key = "ObjectType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ObjectTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to ObjectTypeNode.
 * \sa ObjectTypeNode.
 */
class ObjectType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param span The span of the type.
   */
  MATX_DLL explicit ObjectType(bool is_view = false, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ObjectType, Type, ObjectTypeNode);
};

class StringTypeNode : public TypeNode {
 public:
  bool is_view = false;

  StringTypeNode() {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("is_view", &is_view);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const StringTypeNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  bool Iterable() const override {
    return true;
  }

  bool HasBeginEnd() const override {
    return true;
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"bytes";
  }

  static constexpr const char* _type_key = "StringType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(StringTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to StringTypeNode.
 * \sa StringTypeNode.
 */
class StringType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param span The span of the type.
   */
  MATX_DLL explicit StringType(bool is_view = false, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StringType, Type, StringTypeNode);
};

class UnicodeTypeNode : public TypeNode {
 public:
  bool is_view = false;

  UnicodeTypeNode() {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("is_view", &is_view);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const UnicodeTypeNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  bool Iterable() const override {
    return true;
  }

  bool HasBeginEnd() const override {
    return true;
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"str";
  }

  static constexpr const char* _type_key = "UnicodeType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(UnicodeTypeNode, TypeNode);
};

class UnicodeType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param span The span of the type.
   */
  MATX_DLL explicit UnicodeType(bool is_view = false, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(UnicodeType, Type, UnicodeTypeNode);
};

class ListTypeNode : public TypeNode {
 public:
  /*! \brief The type of item in the List. */
  Type item_type;
  bool is_full_typed = false;

  ListTypeNode() {
  }

  bool Iterable() const override {
    return true;
  }

  bool HasBeginEnd() const override {
    return true;
  }

  bool IsFullTyped() const override {
    return is_full_typed;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("item_type", &item_type);
    v->Visit("is_full_typed", &is_full_typed);
  }

  bool SEqualReduce(const ListTypeNode* other, SEqualReducer equal) const {
    return equal(item_type, other->item_type) && equal(is_full_typed, other->is_full_typed);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(item_type);
  }

  runtime::Unicode GetPythonTypeName() const override {
    if (is_full_typed) {
      return U"FTList[" + item_type->GetPythonTypeName() + U"]";
    } else {
      return U"List[" + item_type->GetPythonTypeName() + U"]";
    }
  }

  static constexpr const char* _type_key = "ListType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ListTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to ListTypeNode.
 * \sa ListTypeNode.
 */
class ListType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param item_type The item type of the list.
   * \param span The span of the type.
   */
  MATX_DLL explicit ListType(bool is_full_typed = false,
                             Type item_type = ObjectType(),
                             Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ListType, Type, ListTypeNode);
};

class DictTypeNode : public TypeNode {
 public:
  /*! \brief The key type of item in the List. */
  Type key_type;
  /*! \brief The value type of item in the List. */
  Type value_type;

  bool is_full_typed = false;

  DictTypeNode() {
  }

  bool Iterable() const override {
    return true;
  }

  bool HasBeginEnd() const override {
    return true;
  }

  bool IsFullTyped() const override {
    return is_full_typed;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("key_type", &key_type);
    v->Visit("value_type", &value_type);
    v->Visit("is_full_typed", &is_full_typed);
  }

  bool SEqualReduce(const DictTypeNode* other, SEqualReducer equal) const {
    return equal(key_type, other->key_type) && equal(value_type, other->value_type) &&
           equal(is_full_typed, other->is_full_typed);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(key_type);
    hash_reduce(value_type);
  }

  runtime::Unicode GetPythonTypeName() const override {
    if (is_full_typed) {
      return U"FTDict[" + key_type->GetPythonTypeName() + U", " + value_type->GetPythonTypeName() +
             U"]";
    } else {
      return U"Dict[" + key_type->GetPythonTypeName() + U", " + value_type->GetPythonTypeName() +
             U"]";
    }
  }

  static constexpr const char* _type_key = "DictType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DictTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to DictTypeNode.
 * \sa DictTypeNode.
 */
class DictType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param key_type The key type of the dict.
   * \param value_type The value type of the dict.
   * \param span The span of the type.
   */
  MATX_DLL explicit DictType(bool is_full_typed = false,
                             Type key_type = ObjectType(),
                             Type value_type = ObjectType(),
                             Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DictType, Type, DictTypeNode);
};

class SetTypeNode : public TypeNode {
 public:
  /*! \brief The type of item in the List. */
  Type item_type;

  bool is_full_typed = false;

  SetTypeNode() {
  }

  bool Iterable() const override {
    return true;
  }

  bool HasBeginEnd() const override {
    return true;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("item_type", &item_type);
    v->Visit("is_full_typed", &is_full_typed);
  }

  bool SEqualReduce(const SetTypeNode* other, SEqualReducer equal) const {
    return equal(item_type, other->item_type) && equal(is_full_typed, other->is_full_typed);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(item_type);
  }

  runtime::Unicode GetPythonTypeName() const override {
    if (is_full_typed) {
      return U"FTSet[" + item_type->GetPythonTypeName() + U"]";
    } else {
      return U"Set[" + item_type->GetPythonTypeName() + U"]";
    }
  }

  bool IsFullTyped() const override {
    return is_full_typed;
  }

  static constexpr const char* _type_key = "SetType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SetTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to SetTypeNode.
 * \sa SetTypeNode.
 */
class SetType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param item_type The item type of the set.
   * \param span The span of the type.
   */
  MATX_DLL explicit SetType(bool is_full_typed = false,
                            Type item_type = ObjectType(),
                            Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SetType, Type, SetTypeNode);
};

class IteratorTypeNode : public TypeNode {
 public:
  /*!
   * \brief The type of the container which gen the iterator.
   */
  Type container_type;
  Type value_type;
  bool has_begin_end = false;

  bool Iterable() const override {
    return true;
  }

  bool HasBeginEnd() const override {
    return has_begin_end;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("container_type", &container_type);
    v->Visit("value_type", &value_type);
    v->Visit("has_begin_end", &has_begin_end);
  }

  bool SEqualReduce(const IteratorTypeNode* other, SEqualReducer equal) const {
    return equal(container_type, other->container_type);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(container_type);
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"iterator";
  }

  static constexpr const char* _type_key = "IteratorType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IteratorTypeNode, TypeNode);
};

/*
 * \brief Managed reference to IteratorTypeNode.
 * \sa IteratorTypeNode
 */
class IteratorType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param container_type The type of the container which gen the Iter.
   * \param value_type The type of the container which gen the Iter.
   */
  MATX_DLL explicit IteratorType(Type container_type, Span span = Span());
  MATX_DLL explicit IteratorType(Type container_type,
                                 Type value_type,
                                 bool has_begin_end,
                                 Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(IteratorType, Type, IteratorTypeNode);
};

class ExceptionTypeNode : public TypeNode {
 public:
  StringRef name;

  ExceptionTypeNode() {
  }
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("span", &span);
  }
  bool SEqualReduce(const ExceptionTypeNode* other, SEqualReducer equal) const {
    return equal(name, other->name);
  }
  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
  }

  runtime::Unicode GetPythonTypeName() const override {
    return runtime::StringHelper::Decode(name);
  }

  static constexpr const char* _type_key = "ExceptionType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ExceptionTypeNode, TypeNode);
};

class ExceptionType : public Type {
 public:
  MATX_DLL explicit ExceptionType(StringRef name, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExceptionType, Type, ExceptionTypeNode);
};

class FileTypeNode : public TypeNode {
 public:
  bool binary_mode = false;

  FileTypeNode() {
  }

  bool Iterable() const override {
    return true;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("binary_mode", &binary_mode);
  }

  bool SEqualReduce(const FileTypeNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"file";
  }

  static constexpr const char* _type_key = "FileType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FileTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to FileTypeNode.
 * \sa FileTypeNode.
 */
class FileType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param span The span of the type.
   */
  MATX_DLL explicit FileType(bool binary_mode, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(FileType, Type, FileTypeNode);
};

class TrieTypeNode : public TypeNode {
 public:
  TrieTypeNode() {
  }
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
  }
  bool SEqualReduce(const TrieTypeNode* other, SEqualReducer equal) const {
    return true;
  }
  void SHashReduce(SHashReducer hash_reduce) const {
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"Trie";
  }

  static constexpr const char* _type_key = "TrieType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TrieTypeNode, TypeNode);
};
class TrieType : public Type {
 public:
  MATX_DLL explicit TrieType(Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TrieType, Type, TrieTypeNode);
};

class UserDataTypeNode : public TypeNode {
 public:
  UserDataTypeNode() {
  }
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
  }
  bool SEqualReduce(const UserDataTypeNode* other, SEqualReducer equal) const {
    return true;
  }
  void SHashReduce(SHashReducer hash_reduce) const {
  }

  static constexpr const char* _type_key = "UserDataType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(UserDataTypeNode, TypeNode);
};
class UserDataType : public Type {
 public:
  MATX_DLL explicit UserDataType(Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(UserDataType, Type, UserDataTypeNode);
};

class NDArrayTypeNode : public TypeNode {
 public:
  int64_t ndim = -1;  // -1 means unknown
  PrimType dtype;     // unknown if not defined
  NDArrayTypeNode() {
  }
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("ndim", &ndim);
    v->Visit("dtype", &dtype);
    v->Visit("span", &span);
  }
  bool SEqualReduce(const NDArrayTypeNode* other, SEqualReducer equal) const {
    return (ndim == other->ndim) && (equal(dtype, other->dtype));
  }
  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(ndim);
    hash_reduce(dtype);
  }

  bool Iterable() const override {
    return true;
  }

  runtime::Unicode GetPythonTypeName() const override;

  static constexpr const char* _type_key = "NDArrayType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(NDArrayTypeNode, TypeNode);
};

class NDArrayType : public Type {
 public:
  MATX_DLL explicit NDArrayType(int64_t ndim, PrimType dtype, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(NDArrayType, Type, NDArrayTypeNode);
};

class RegexTypeNode : public TypeNode {
 public:
  RegexTypeNode() {
  }
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
  }
  bool SEqualReduce(const RegexTypeNode* other, SEqualReducer equal) const {
    return true;
  }
  void SHashReduce(SHashReducer hash_reduce) const {
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"Regex";
  }

  static constexpr const char* _type_key = "RegexType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(RegexTypeNode, TypeNode);
};

class RegexType : public Type {
 public:
  MATX_DLL explicit RegexType(Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RegexType, Type, RegexTypeNode);
};

class OpaqueObjectTypeNode : public TypeNode {
 public:
  OpaqueObjectTypeNode() {
  }
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
  }
  bool SEqualReduce(const OpaqueObjectTypeNode* other, SEqualReducer equal) const {
    return true;
  }
  void SHashReduce(SHashReducer hash_reduce) const {
  }

  static constexpr const char* _type_key = "OpaqueObjectType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(OpaqueObjectTypeNode, TypeNode);
};

class OpaqueObjectType : public Type {
 public:
  MATX_DLL explicit OpaqueObjectType(Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(OpaqueObjectType, Type, OpaqueObjectTypeNode);
};

class RefTypeNode : public TypeNode {
 public:
  /*! \brief The type of value in the Reference. */
  Type value;

  RefTypeNode() {
  }

  bool Iterable() const override {
    return value->Iterable();
  }

  bool HasBeginEnd() const override {
    return value->HasBeginEnd();
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const RefTypeNode* other, SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(value);
  }

  runtime::Unicode GetPythonTypeName() const override {
    return U"Ref[" + value->GetPythonTypeName() + U"]";
  }

  static constexpr const char* _type_key = "RefType";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(RefTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to RefTypeNode.
 * \sa RefTypeNode.
 */
class RefType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param value The type of value in the reference.
   * \param span The span of the type.
   */
  MATX_DLL explicit RefType(Type value, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RefType, Type, RefTypeNode);
};

inline bool IsPrimType(const Type& t) {
  return t->IsInstance<PrimTypeNode>();
}

inline bool IsIntegerType(const Type& t) {
  if (auto* t_node = t.as<PrimTypeNode>()) {
    return t_node->dtype.is_int() || t_node->dtype.is_uint() || t_node->dtype.is_bool();
  }
  return false;
}

inline bool IsFloatType(const Type& t) {
  return t->IsInstance<PrimTypeNode>() && GetRuntimeDataType(t).is_float();
}

inline bool IsStringType(const Type& t) {
  return t->IsInstance<StringTypeNode>();
}

inline bool IsUnicodeType(const Type& t) {
  return t->IsInstance<UnicodeTypeNode>();
}

inline bool IsListType(const Type& t) {
  return t->IsInstance<ListTypeNode>();
}

inline bool IsDictType(const Type& t) {
  return t->IsInstance<DictTypeNode>();
}

inline bool IsSetType(const Type& t) {
  return t->IsInstance<SetTypeNode>();
}

inline bool IsTupleType(const Type& t) {
  return t->IsInstance<TupleTypeNode>();
}

inline bool IsObjectType(const Type& t) {
  return t->IsInstance<ObjectTypeNode>();
}

inline bool IsIteratorType(const Type& t) {
  return t->IsInstance<IteratorTypeNode>();
}

inline bool IsFileType(const Type& t) {
  return t->IsInstance<FileTypeNode>();
}

inline bool IsTrieType(const Type& t) {
  return t->IsInstance<TrieTypeNode>();
}

inline bool IsUserDataType(const Type& t) {
  return t->IsInstance<UserDataTypeNode>();
}

inline bool IsNDArrayType(const Type& t) {
  return t->IsInstance<NDArrayTypeNode>();
}

inline bool IsOpaqueObjectType(const Type& t) {
  return t->IsInstance<OpaqueObjectTypeNode>();
}

inline bool IsRefType(const Type& t) {
  return t->IsInstance<RefTypeNode>();
}

inline const Type& RemoveReference(const Type& t) {
  if (auto* node = t.as<RefTypeNode>()) {
    return RemoveReference(node->value);
  }
  return t;
}

Type InferIteratorValueType(const Type& cons_ty);

Type InferNthItemType(const Type& cons_ty, int64_t index);

bool IsTypeConvertible(const Type& from, const Type& to);

Type InferLiftType(const Type& t1, const Type& t2);

}  // namespace ir
}  // namespace matxscript
