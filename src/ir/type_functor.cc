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
 * \file type_functor.cc
 * \brief Implementations of type functors.
 */
#include <matxscript/ir/type_functor.h>

#include <utility>

#include <matxscript/ir/printer/text_printer.h>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::GetRef;

/******************************************************************************
 * TypeVisitor
 *****************************************************************************/

void TypeVisitor::VisitType_(const TypeVarNode* op) {
}

void TypeVisitor::VisitType_(const FuncTypeNode* op) {
  for (auto type_param : op->type_params) {
    this->VisitType(type_param);
  }

  for (auto type_cs : op->type_constraints) {
    this->VisitType(type_cs);
  }

  for (auto arg_type : op->arg_types) {
    this->VisitType(arg_type);
  }
  this->VisitType(op->ret_type);
}

void TypeVisitor::VisitType_(const RangeTypeNode* op) {
}

void TypeVisitor::VisitType_(const TupleTypeNode* op) {
  for (const Type& t : op->fields) {
    this->VisitType(t);
  }
}

void TypeVisitor::VisitType_(const GlobalTypeVarNode* op) {
}

void TypeVisitor::VisitType_(const PrimTypeNode* op) {
}

void TypeVisitor::VisitType_(const PointerTypeNode* op) {
  this->VisitType(op->element_type);
}

void TypeVisitor::VisitType_(const ObjectTypeNode* op) {
}

void TypeVisitor::VisitType_(const StringTypeNode* op) {
}

void TypeVisitor::VisitType_(const UnicodeTypeNode* op) {
}

void TypeVisitor::VisitType_(const ListTypeNode* op) {
  this->VisitType(op->item_type);
}

void TypeVisitor::VisitType_(const DictTypeNode* op) {
  this->VisitType(op->key_type);
  this->VisitType(op->value_type);
}

void TypeVisitor::VisitType_(const SetTypeNode* op) {
  this->VisitType(op->item_type);
}

void TypeVisitor::VisitType_(const IteratorTypeNode* op) {
  this->VisitType(op->container_type);
}

void TypeVisitor::VisitType_(const ExceptionTypeNode* op) {
}

void TypeVisitor::VisitType_(const FileTypeNode* op) {
}

void TypeVisitor::VisitType_(const NDArrayTypeNode* op) {
  if (op->dtype.defined()) {
    this->VisitType(op->dtype);
  }
}

void TypeVisitor::VisitType_(const ClassTypeNode* op) {
  this->VisitType(op->header);
  for (auto& ty : op->var_types) {
    this->VisitType(ty);
  }
  for (auto& ty : op->func_types) {
    this->VisitType(ty);
  }
  if (op->base.defined()) {
    this->VisitType(op->base);
  }
}

void TypeVisitor::VisitType_(const UserDataTypeNode* op) {
}

void TypeVisitor::VisitType_(const OpaqueObjectTypeNode* op) {
}

void TypeVisitor::VisitType_(const RefTypeNode* op) {
  this->VisitType(op->value);
}

/******************************************************************************
 * TypeMutator
 *****************************************************************************/

Type TypeMutator::VisitType(const Type& t) {
  return t.defined() ? TypeFunctor<Type(const Type&)>::VisitType(t) : t;
}

// Type Mutator.
Array<Type> TypeMutator::MutateArray(Array<Type> arr) {
  // The array will do copy on write
  // If no changes are made, the original array will be returned.
  for (size_t i = 0; i < arr.size(); ++i) {
    Type ty = arr[i];
    Type new_ty = VisitType(ty);
    if (!ty.same_as(new_ty)) {
      arr.Set(i, new_ty);
    }
  }
  return arr;
}

Type TypeMutator::VisitType_(const TypeVarNode* op) {
  return GetRef<TypeVar>(op);
}

Type TypeMutator::VisitType_(const FuncTypeNode* op) {
  bool changed = false;
  Array<TypeVar> type_params;
  for (auto type_param : op->type_params) {
    auto new_type_param = VisitType(type_param);
    changed = changed || !new_type_param.same_as(type_param);
    if (const TypeVarNode* tin = new_type_param.as<TypeVarNode>()) {
      type_params.push_back(GetRef<TypeVar>(tin));
    } else {
      MXLOG(FATAL) << new_type_param;
    }
  }

  Array<TypeConstraint> type_constraints;
  for (auto type_cs : op->type_constraints) {
    auto new_type_cs = VisitType(type_cs);
    changed = changed || !new_type_cs.same_as(type_cs);
    if (const TypeConstraintNode* tin = new_type_cs.as<TypeConstraintNode>()) {
      type_constraints.push_back(GetRef<TypeConstraint>(tin));
    } else {
      MXLOG(FATAL) << new_type_cs;
    }
  }

  Array<Type> new_args = MutateArray(op->arg_types);
  changed = changed || !new_args.same_as(op->arg_types);

  Type new_ret_type = VisitType(op->ret_type);
  changed = changed || !new_ret_type.same_as(op->ret_type);

  if (!changed)
    return GetRef<Type>(op);
  return FuncType(new_args, new_ret_type, type_params, type_constraints);
}

Type TypeMutator::VisitType_(const RangeTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const TupleTypeNode* op) {
  Array<Type> new_fields = MutateArray(op->fields);
  if (new_fields.same_as(op->fields)) {
    return GetRef<Type>(op);
  } else {
    return TupleType(new_fields, op->is_std_tuple, op->span);
  }
}

Type TypeMutator::VisitType_(const GlobalTypeVarNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const PrimTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const PointerTypeNode* op) {
  Type element_type = VisitType(op->element_type);

  if (element_type.same_as(op->element_type)) {
    return GetRef<Type>(op);
  } else {
    return PointerType(element_type);
  }
}

Type TypeMutator::VisitType_(const ObjectTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const StringTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const UnicodeTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const ListTypeNode* op) {
  auto item_type = this->VisitType(op->item_type);
  if (op->item_type.same_as(item_type)) {
    return GetRef<Type>(op);
  }
  return ListType(op->is_full_typed, item_type);
}

Type TypeMutator::VisitType_(const DictTypeNode* op) {
  auto key_type = this->VisitType(op->key_type);
  auto value_type = this->VisitType(op->value_type);
  if (op->key_type.same_as(key_type) && op->value_type.same_as(value_type)) {
    return GetRef<Type>(op);
  }
  return DictType(op->is_full_typed, key_type, value_type);
}

Type TypeMutator::VisitType_(const SetTypeNode* op) {
  auto item_type = this->VisitType(op->item_type);
  if (op->item_type.same_as(item_type)) {
    return GetRef<Type>(op);
  }
  return SetType(op->is_full_typed, item_type);
}

Type TypeMutator::VisitType_(const IteratorTypeNode* op) {
  Type container_type = VisitType(op->container_type);
  if (container_type.same_as(op->container_type)) {
    return GetRef<Type>(op);
  } else {
    return PointerType(container_type);
  }
}

Type TypeMutator::VisitType_(const ExceptionTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const FileTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const NDArrayTypeNode* op) {
  if (op->dtype.defined()) {
    auto dtype = this->VisitType(op->dtype);
    if (!op->dtype.same_as(dtype)) {
      return NDArrayType(op->ndim, op->dtype, op->span);
    }
  }
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const ClassTypeNode* op) {
  // var names and function names can't be changed by default
  bool changed = false;
  GlobalTypeVar header = runtime::Downcast<GlobalTypeVar>(VisitType(op->header));
  changed &= header.same_as(op->header);
  Array<Type> var_types;
  for (auto& ty : op->var_types) {
    auto nty = this->VisitType(ty);
    var_types.push_back(nty);
    changed &= nty.same_as(ty);
  }
  Array<FuncType> func_types;
  for (auto& ty : op->func_types) {
    auto nty = runtime::Downcast<FuncType>(this->VisitType(ty));
    func_types.push_back(nty);
    changed &= nty.same_as(ty);
  }
  Type base;
  if (op->base.defined()) {
    base = this->VisitType(op->base);
    changed &= base.same_as(op->base);
  }
  if (changed) {
    auto new_ty = ClassType(op->py_type_id,
                            std::move(header),
                            std::move(base),
                            op->var_names,
                            var_types,
                            op->func_names,
                            op->unbound_func_names,
                            std::move(func_types));
    ((ClassTypeNode*)new_ty.get())->tag = op->tag;
    return new_ty;
  } else {
    return GetRef<Type>(op);
  }
}

Type TypeMutator::VisitType_(const UserDataTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const OpaqueObjectTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const RefTypeNode* op) {
  auto value = this->VisitType(op->value);
  if (op->value.same_as(value)) {
    return GetRef<Type>(op);
  }
  return RefType(value);
}

// Implements bind.
class TypeBinder : public TypeMutator {
 public:
  explicit TypeBinder(const Map<TypeVar, Type>& args_map) : args_map_(args_map) {
  }

  Type VisitType_(const TypeVarNode* op) override {
    auto id = GetRef<TypeVar>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return std::move(id);
    }
  }

 private:
  const Map<TypeVar, Type>& args_map_;
};

Type Bind(const Type& type, const Map<TypeVar, Type>& args_map) {
  return TypeBinder(args_map).VisitType(type);
}

}  // namespace ir
}  // namespace matxscript
