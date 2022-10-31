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
 * \file matx/ir/type_functor.h
 * \brief A way to defined arbitrary function signature with dispatch on types.
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <matxscript/ir/adt.h>
#include <matxscript/ir/base.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::NodeFunctor;

template <typename FType>
class TypeFunctor;

// functions to be overridden.
#define MATXSCRIPT_TYPE_FUNCTOR_DEFAULT \
  { return VisitTypeDefault_(op, std::forward<Args>(args)...); }

#define MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(OP)                                               \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitType_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class TypeFunctor<R(const Type& n, Args...)> {
 private:
  using TSelf = TypeFunctor<R(const Type& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~TypeFunctor() {
  }
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Type& n, Args... args) {
    return VisitType(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitType(const Type& n, Args... args) {
    MXCHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitType_(const TypeVarNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeConstraintNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FuncTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TupleTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const GlobalTypeVarNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PrimTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PointerTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;

  virtual R VisitType_(const ObjectTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const UnicodeTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const StringTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ListTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const DictTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const SetTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const IteratorTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ExceptionTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FileTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const NDArrayTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ClassTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const UserDataTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const OpaqueObjectTypeNode* op,
                       Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const RefTypeNode* op, Args... args) MATXSCRIPT_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitTypeDefault_(const Object* op, Args...) {
    MXLOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;  // unreachable, written to stop compiler warning
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(TypeVarNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(TypeConstraintNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(FuncTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(TupleTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(GlobalTypeVarNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(PrimTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(PointerTypeNode);

    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(ObjectTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(UnicodeTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(StringTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(ListTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(DictTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(SetTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(IteratorTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(ExceptionTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(FileTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(NDArrayTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(ClassTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(UserDataTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(OpaqueObjectTypeNode);
    MATXSCRIPT_TYPE_FUNCTOR_DISPATCH(RefTypeNode);
    return vtable;
  }
};

#undef MATXSCRIPT_TYPE_FUNCTOR_DISPATCH

/*!
 * \brief A type visitor that recursively visit types.
 */
class MATX_DLL TypeVisitor : public TypeFunctor<void(const Type& n)> {
 public:
  void VisitType_(const TypeVarNode* op) override;
  void VisitType_(const FuncTypeNode* op) override;
  void VisitType_(const TupleTypeNode* op) override;
  void VisitType_(const GlobalTypeVarNode* op) override;
  void VisitType_(const PrimTypeNode* op) override;
  void VisitType_(const PointerTypeNode* op) override;

  void VisitType_(const ObjectTypeNode* op) override;
  void VisitType_(const UnicodeTypeNode* op) override;
  void VisitType_(const StringTypeNode* op) override;
  void VisitType_(const ListTypeNode* op) override;
  void VisitType_(const DictTypeNode* op) override;
  void VisitType_(const SetTypeNode* op) override;
  void VisitType_(const ExceptionTypeNode* op) override;
  void VisitType_(const IteratorTypeNode* op) override;
  void VisitType_(const FileTypeNode* op) override;
  void VisitType_(const NDArrayTypeNode* op) override;
  void VisitType_(const ClassTypeNode* op) override;
  void VisitType_(const UserDataTypeNode* op) override;
  void VisitType_(const OpaqueObjectTypeNode* op) override;
  void VisitType_(const RefTypeNode* op) override;
};

/*!
 * \brief TypeMutator that mutates expressions.
 */
class MATX_DLL TypeMutator : public TypeFunctor<Type(const Type& n)> {
 public:
  Type VisitType(const Type& t) override;
  Type VisitType_(const TypeVarNode* op) override;
  Type VisitType_(const FuncTypeNode* op) override;
  Type VisitType_(const TupleTypeNode* op) override;
  Type VisitType_(const GlobalTypeVarNode* op) override;
  Type VisitType_(const PrimTypeNode* op) override;
  Type VisitType_(const PointerTypeNode* op) override;

  Type VisitType_(const ObjectTypeNode* op) override;
  Type VisitType_(const UnicodeTypeNode* op) override;
  Type VisitType_(const StringTypeNode* op) override;
  Type VisitType_(const ListTypeNode* op) override;
  Type VisitType_(const DictTypeNode* op) override;
  Type VisitType_(const SetTypeNode* op) override;
  Type VisitType_(const ExceptionTypeNode* op) override;
  Type VisitType_(const IteratorTypeNode* op) override;
  Type VisitType_(const FileTypeNode* op) override;
  Type VisitType_(const NDArrayTypeNode* op) override;
  Type VisitType_(const ClassTypeNode* op) override;
  Type VisitType_(const UserDataTypeNode* op) override;
  Type VisitType_(const OpaqueObjectTypeNode* op) override;
  Type VisitType_(const RefTypeNode* op) override;

 private:
  runtime::Array<Type> MutateArray(runtime::Array<Type> arr);
};

/*!
 * \brief Bind free type variables in the type.
 * \param type The type to be updated.
 * \param args_map The binding map.
 */
Type Bind(const Type& type, const runtime::Map<TypeVar, Type>& args_map);

}  // namespace ir
}  // namespace matxscript
