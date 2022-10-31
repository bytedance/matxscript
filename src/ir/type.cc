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
 * \file src/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */
#include <matxscript/ir/type.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/adt.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;

bool IsRuntimeDataType(const Type& type) {
  if (auto* n = type.as<PrimTypeNode>()) {
    return true;
  } else if (type.as<PointerTypeNode>()) {
    return true;
  } else if (IsVoidType(type)) {
    return true;
  } else {
    return false;
  }
}

runtime::DataType GetRuntimeDataType(const Type& type) {
  if (auto* n = type.as<PrimTypeNode>()) {
    return n->dtype;
  } else if (type.as<PointerTypeNode>()) {
    return DataType::Handle();
  } else if (IsVoidType(type)) {
    return DataType::Void();
  } else {
    MXLOG(FATAL) << "Type " << type << " does not have a corresponding runtime::DataType";
    return DataType::Handle();
  }
}

PrimType::PrimType(runtime::DataType dtype) {
  ObjectPtr<PrimTypeNode> n = make_object<PrimTypeNode>();
  n->dtype = dtype;
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(PrimTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimType").set_body_typed([](runtime::DataType dtype) {
  return PrimType(dtype);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimType_GetDType").set_body_typed([](PrimType pt) {
  return pt->dtype;
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PrimTypeNode*>(ref.get());
      p->stream << node->dtype;
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.VoidType").set_body_typed([]() { return VoidType(); });
MATXSCRIPT_REGISTER_GLOBAL("ir.IsVoidType").set_body_typed([](const Type& type) {
  return IsVoidType(type);
});

PointerType::PointerType(Type element_type) {
  ObjectPtr<PointerTypeNode> n = make_object<PointerTypeNode>();
  n->element_type = std::move(element_type);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(PointerTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.PointerType").set_body_typed([](Type element_type) {
  return PointerType(element_type);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PointerTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PointerTypeNode*>(ref.get());
      p->Print(node->element_type);
      p->stream << '*';
    });

TypeVar::TypeVar(StringRef name, TypeKind kind, Span span) {
  ObjectPtr<TypeVarNode> n = make_object<TypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(TypeVarNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.TypeVar").set_body_typed([](StringRef name, int kind) {
  return TypeVar(name, static_cast<TypeKind>(kind));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TypeVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TypeVarNode*>(ref.get());
      p->stream << "TypeVar(" << node->name_hint << ", " << node->kind << ")";
    });

GlobalTypeVar::GlobalTypeVar(StringRef name, TypeKind kind, Span span) {
  ObjectPtr<GlobalTypeVarNode> n = make_object<GlobalTypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(GlobalTypeVarNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.GlobalTypeVar").set_body_typed([](StringRef name, int kind) {
  return GlobalTypeVar(name, static_cast<TypeKind>(kind));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<GlobalTypeVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const GlobalTypeVarNode*>(ref.get());
      p->stream << "GlobalTypeVar(" << node->name_hint << ", " << node->kind << ")";
    });

FuncType::FuncType(runtime::Array<Type> arg_types,
                   Type ret_type,
                   runtime::Array<TypeVar> type_params,
                   runtime::Array<TypeConstraint> type_constraints,
                   Span span) {
  ObjectPtr<FuncTypeNode> n = make_object<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->type_constraints = std::move(type_constraints);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(FuncTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.FuncType")
    .set_body_typed([](runtime::Array<Type> arg_types,
                       Type ret_type,
                       runtime::Array<TypeVar> type_params,
                       runtime::Array<TypeConstraint> type_constraints) {
      return FuncType(std::move(arg_types),
                      std::move(ret_type),
                      std::move(type_params),
                      std::move(type_constraints));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FuncTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FuncTypeNode*>(ref.get());
      p->stream << "FuncType(" << node->type_params << ", " << node->arg_types << ", "
                << node->ret_type << ", " << node->type_constraints << ")";
    });

TupleType::TupleType(runtime::Array<Type> fields, bool is_std_tuple, Span span) {
  ObjectPtr<TupleTypeNode> n = make_object<TupleTypeNode>();
  n->fields = std::move(fields);
  n->is_std_tuple = is_std_tuple;
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleType TupleType::Empty() {
  return TupleType(runtime::Array<Type>(), false);
}

MATXSCRIPT_REGISTER_NODE_TYPE(TupleTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.TupleType").set_body_typed([](runtime::Array<Type> fields) {
  return TupleType(std::move(fields));
});

MATXSCRIPT_REGISTER_GLOBAL("ir.TupleType_Len").set_body_typed([](TupleType ty) {
  return (int64_t)ty->fields.size();
});

MATXSCRIPT_REGISTER_GLOBAL("ir.TupleType_GetItem").set_body_typed([](TupleType ty, int64_t index) {
  return ty->fields[index];
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleTypeNode*>(ref.get());
      p->stream << "TupleTypeNode(" << node->fields << ")";
    });

// Object Type
ObjectType::ObjectType(bool is_view, Span span) {
  ObjectPtr<ObjectTypeNode> n = make_object<ObjectTypeNode>();
  n->is_view = is_view;
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ObjectTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ObjectType").set_body_typed([](bool is_view) {
  return ObjectType(is_view);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ObjectTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ObjectTypeNode*>(ref.get());
      p->stream << "ObjectTypeNode";
    });

// String Type
StringType::StringType(bool is_view, Span span) {
  ObjectPtr<StringTypeNode> n = make_object<StringTypeNode>();
  n->span = std::move(span);
  n->is_view = is_view;
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(StringTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.StringType").set_body_typed([](bool is_view) {
  return StringType(is_view);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StringTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const StringTypeNode*>(ref.get());
      p->stream << "StringTypeNode";
    });

// Unicode Type
UnicodeType::UnicodeType(bool is_view, Span span) {
  ObjectPtr<UnicodeTypeNode> n = make_object<UnicodeTypeNode>();
  n->span = std::move(span);
  n->is_view = is_view;
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(UnicodeTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.UnicodeType").set_body_typed([](bool is_view) {
  return UnicodeType(is_view);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UnicodeTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const UnicodeTypeNode*>(ref.get());
      p->stream << "UnicodeTypeNode";
    });

// List Type
ListType::ListType(bool is_full_typed, Type item_type, Span span) {
  ObjectPtr<ListTypeNode> n = make_object<ListTypeNode>();
  n->item_type = std::move(item_type);
  n->span = std::move(span);
  n->is_full_typed = is_full_typed;
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ListTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ListType").set_body_typed([](bool is_full_typed, Type item_type) {
  return ListType(is_full_typed, std::move(item_type));
});
MATXSCRIPT_REGISTER_GLOBAL("ir.ListTypeGetItemType").set_body_typed([](ListType t) {
  return t->item_type;
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ListTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ListTypeNode*>(ref.get());
      p->stream << "ListTypeNode(" << node->item_type << ")";
    });

// Dict Type
DictType::DictType(bool is_full_typed, Type key_type, Type value_type, Span span) {
  ObjectPtr<DictTypeNode> n = make_object<DictTypeNode>();
  n->key_type = std::move(key_type);
  n->value_type = std::move(value_type);
  n->span = std::move(span);
  n->is_full_typed = is_full_typed;
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(DictTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.DictType")
    .set_body_typed([](bool is_full_typed, Type key_type, Type value_type) {
      return DictType(is_full_typed, std::move(key_type), std::move(value_type));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DictTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DictTypeNode*>(ref.get());
      p->stream << "DictTypeNode(key_type: " << node->key_type
                << ", value_type: " << node->value_type << ")";
    });

// Set Type
SetType::SetType(bool is_full_typed, Type item_type, Span span) {
  ObjectPtr<SetTypeNode> n = make_object<SetTypeNode>();
  n->span = std::move(span);
  n->item_type = std::move(item_type);
  n->is_full_typed = is_full_typed;
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(SetTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.SetType").set_body_typed([](bool is_full_typed, Type item_type) {
  return SetType(is_full_typed, std::move(item_type));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SetTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const SetTypeNode*>(ref.get());
      p->stream << "SetTypeNode(" << node->item_type << ")";
    });

// IteratorType
IteratorType::IteratorType(Type container_type, Span span) {
  ObjectPtr<IteratorTypeNode> n = make_object<IteratorTypeNode>();
  n->value_type = InferIteratorValueType(container_type);
  n->has_begin_end = false;
  n->container_type = std::move(container_type);
  n->span = std::move(span);
  data_ = std::move(n);
}

IteratorType::IteratorType(Type container_type, Type value_type, bool has_begin_end, Span span) {
  ObjectPtr<IteratorTypeNode> n = make_object<IteratorTypeNode>();
  n->value_type = std::move(value_type);
  n->has_begin_end = has_begin_end;
  n->container_type = std::move(container_type);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(IteratorTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.IteratorType").set_body_typed([](Type container_type) {
  return IteratorType(std::move(container_type));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IteratorTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IteratorTypeNode*>(ref.get());
      p->Print(node->container_type);
      p->stream << "_Iterator";
    });

// ExceptionType
ExceptionType::ExceptionType(StringRef name, Span span) {
  ObjectPtr<ExceptionTypeNode> n = make_object<ExceptionTypeNode>();
  n->name = std::move(name);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ExceptionTypeNode);
MATXSCRIPT_REGISTER_GLOBAL("ir.ExceptionType").set_body_typed([](StringRef name) {
  return ExceptionType(std::move(name));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ExceptionTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ExceptionTypeNode*>(ref.get());
      p->Print(node->name);
    });

// FileType
FileType::FileType(bool binary_mode, Span span) {
  ObjectPtr<FileTypeNode> n = make_object<FileTypeNode>();
  n->binary_mode = binary_mode;
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(FileTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.FileType").set_body_typed([](bool binary_mode) {
  return FileType(binary_mode);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FileTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FileTypeNode*>(ref.get());
      p->stream << "FileTypeNode(binary_mode=" << node->binary_mode << ")";
    });

// TrieType
TrieType::TrieType(Span span) {
  ObjectPtr<TrieTypeNode> n = make_object<TrieTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(TrieTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.TrieType").set_body_typed([]() { return TrieType(); });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TrieTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "TrieType";
    });

// UserDataType
UserDataType::UserDataType(Span span) {
  ObjectPtr<UserDataTypeNode> n = make_object<UserDataTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(UserDataTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.UserDataType").set_body_typed([]() { return UserDataType(); });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UserDataTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "UserDataType";
    });

// NDArrayType
runtime::Unicode NDArrayTypeNode::GetPythonTypeName() const {
  std::stringstream os;
  os << "NDArray[ndim=";
  if (ndim < 0) {
    os << "?";
  } else {
    os << ndim;
  }
  os << ", dtype=";
  if (dtype.defined()) {
    os << dtype->dtype;
  } else {
    os << "?";
  }
  os << "]";
  auto s = os.str();
  return runtime::String(s.data(), s.size()).decode();
}

NDArrayType::NDArrayType(int64_t ndim, PrimType dtype, Span span) {
  ObjectPtr<NDArrayTypeNode> n = make_object<NDArrayTypeNode>();
  n->ndim = ndim;
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(NDArrayTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.NDArrayType").set_body_typed([](int64_t ndim, PrimType dtype) {
  return NDArrayType(ndim, std::move(dtype));
});

RegexType::RegexType(Span span) {
  ObjectPtr<RegexTypeNode> n = make_object<RegexTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(RegexTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.RegexType").set_body_typed([]() { return RegexType(); });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RegexTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "RegexType";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<NDArrayTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "NDArrayType";
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.Type_GetPythonTypeName").set_body_typed([](Type ty) {
  return ty->GetPythonTypeName();
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Type_IsFullTyped").set_body_typed([](Type ty) {
  return ty->IsFullTyped();
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Type_IsIterable").set_body_typed([](Type ty) {
  return ty->Iterable();
});

// OpaqueObjectType
OpaqueObjectType::OpaqueObjectType(Span span) {
  ObjectPtr<OpaqueObjectTypeNode> n = make_object<OpaqueObjectTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(OpaqueObjectTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.OpaqueObjectType").set_body_typed([]() {
  return OpaqueObjectType();
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<OpaqueObjectTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "OpaqueObjectType";
    });

// Ref Type
RefType::RefType(Type value, Span span) {
  ObjectPtr<RefTypeNode> n = make_object<RefTypeNode>();
  n->span = std::move(span);
  n->value = std::move(value);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(RefTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.RefType").set_body_typed([](Type value) {
  return RefType(std::move(value));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefTypeNode*>(ref.get());
      p->stream << "RefTypeNode(" << node->value << ")";
    });

Type InferIteratorValueType(const Type& cons_ty) {
  if (auto* ptr = cons_ty.as<ListTypeNode>()) {
    return ptr->item_type;
  } else if (auto* ptr = cons_ty.as<SetTypeNode>()) {
    return ptr->item_type;
  } else if (auto* ptr = cons_ty.as<DictTypeNode>()) {
    return ptr->key_type;
  } else if (auto* ptr = cons_ty.as<StringTypeNode>()) {
    return PrimType(runtime::DataType::Int(64));
  } else if (auto* ptr = cons_ty.as<UnicodeTypeNode>()) {
    return UnicodeType();
  } else if (auto* ptr = cons_ty.as<IteratorTypeNode>()) {
    return ptr->value_type;
  } else if (auto* ptr = cons_ty.as<TupleTypeNode>()) {
    Type item_type = ObjectType();
    if (!ptr->fields.empty()) {
      bool is_same = true;
      auto& f_ty_0 = ptr->fields[0];
      if (f_ty_0.defined()) {
        for (auto i = 1; i < ptr->fields.size(); ++i) {
          auto& f_ty_i = ptr->fields[i];
          if ((!f_ty_i.defined()) || f_ty_0->type_index() != ptr->fields->type_index()) {
            is_same = false;
          }
        }
      }
      if (is_same) {
        item_type = f_ty_0;
      }
    }
    return item_type;
  } else if (auto* ptr = cons_ty.as<FileTypeNode>()) {
    if (ptr->binary_mode) {
      return StringType();
    }
    return UnicodeType();
  } else if (auto* ptr = cons_ty.as<RefTypeNode>()) {
    return InferIteratorValueType(ptr->value);
  }
  return ObjectType();
}

Type InferNthItemType(const Type& cons_ty, int64_t index) {
  if (auto* ptr = cons_ty.as<TupleTypeNode>()) {
    MXCHECK(index < ptr->fields.size());
    return ptr->fields[index];
  } else if (auto* ptr = cons_ty.as<RefTypeNode>()) {
    return InferNthItemType(ptr->value, index);
  }
  return InferIteratorValueType(cons_ty);
}

bool IsTypeConvertible(const Type& from, const Type& to) {
  if (from.same_as(to)) {
    return true;
  }
  if (const auto* ref_ty = from.as<RefTypeNode>()) {
    return IsTypeConvertible(ref_ty->value, to);
  }
  if (const auto* ref_ty = to.as<RefTypeNode>()) {
    return IsTypeConvertible(from, ref_ty->value);
  }
  if (IsObjectType(from) || IsObjectType(to)) {
    return true;
  }
  if (from->IsFullTyped() ^ to->IsFullTyped()) {
    return false;
  }
  if (IsBaseTypeOf(from, to, true) || IsBaseTypeOf(to, from, true)) {
    return true;
  }
  if (from->IsInstance<ClassTypeNode>()) {
    // return from.same_as(to) || IsUserDataType(to) || IsObjectType(to);
    return IsUserDataType(to);
  }
  if (to->IsInstance<ClassTypeNode>()) {
    // return to.same_as(from) || IsUserDataType(from) || IsObjectType(from);
    return IsUserDataType(from);
  }
  {
    // NDArray type check
    const auto* from_pt = from.as<NDArrayTypeNode>();
    const auto* to_pt = to.as<NDArrayTypeNode>();
    if (from_pt && to_pt) {
      const auto* from_dtype = from_pt->dtype.as<PrimTypeNode>();
      const auto* to_dtype = to_pt->dtype.as<PrimTypeNode>();
      if (from_dtype && to_dtype && (from_dtype->dtype != to_dtype->dtype)) {
        return false;
      }
      if (from_pt->ndim >= 0 && to_pt->ndim >= 0 && (from_pt->ndim != to_pt->ndim)) {
        return false;
      }
      return true;
    }
  }
  {
    // prim type check
    const auto* from_pt = from.as<PrimTypeNode>();
    const auto* to_pt = to.as<PrimTypeNode>();
    if (from_pt && to_pt) {
      if (from_pt->dtype.lanes() != to_pt->dtype.lanes()) {
        return false;
      }
      if (from_pt->dtype.is_float() && !to_pt->dtype.is_float()) {
        return false;
      }
      if (from_pt->dtype.bits() > to_pt->dtype.bits()) {
        return false;
      }
      return true;
    }
  }
  {
    // dict type check
    const auto* from_pt = from.as<DictTypeNode>();
    const auto* to_pt = to.as<DictTypeNode>();
    if (from_pt && to_pt) {
      return IsTypeConvertible(from_pt->key_type, to_pt->key_type) &&
             IsTypeConvertible(from_pt->value_type, to_pt->value_type);
    }
  }
  {
    // tuple type check
    const auto* from_pt = from.as<TupleTypeNode>();
    const auto* to_pt = to.as<TupleTypeNode>();
    if (from_pt && to_pt) {
      if (from_pt->fields.size() == 0 || to_pt->fields.size() == 0) {
        // Generic Tuple[...]
        return true;
      }
      if (from_pt->fields.size() != to_pt->fields.size()) {
        return false;
      }
      for (size_t i = 0; i < from_pt->fields.size(); ++i) {
        if (!IsTypeConvertible(from_pt->fields[i], to_pt->fields[i])) {
          return false;
        }
      }
      return true;
    }
  }
  {
    // list type check
    const auto* from_pt = from.as<ListTypeNode>();
    const auto* to_pt = to.as<ListTypeNode>();
    if (from_pt && to_pt) {
      return IsTypeConvertible(from_pt->item_type, to_pt->item_type);
    }
  }
  {
    // set type check
    const auto* from_pt = from.as<SetTypeNode>();
    const auto* to_pt = to.as<SetTypeNode>();
    if (from_pt && to_pt) {
      return IsTypeConvertible(from_pt->item_type, to_pt->item_type);
    }
  }
  return from == to;
}

Type InferLiftType(const Type& t1, const Type& t2) {
  static ObjectType any_type(false);
  if (t1.same_as(t2)) {
    return t1;
  }
  if (const auto* ref_ty = t1.as<RefTypeNode>()) {
    return InferLiftType(ref_ty->value, t2);
  }
  if (const auto* ref_ty = t2.as<RefTypeNode>()) {
    return InferLiftType(t1, ref_ty->value);
  }
  if (IsObjectType(t1) || IsObjectType(t2)) {
    return any_type;
  }
  if (t1->IsFullTyped() ^ t2->IsFullTyped()) {
    return any_type;
  }
  auto InferPrimType = [&](const PrimTypeNode* prim_ty, const Type& other) -> Type {
    if (auto* other_node = other.as<PrimTypeNode>()) {
      if (prim_ty->dtype == other_node->dtype) {
        return other;
      } else if (prim_ty->dtype.lanes() == other_node->dtype.lanes()) {
        auto bits = prim_ty->dtype.bits() > other_node->dtype.bits() ? prim_ty->dtype.bits()
                                                                     : other_node->dtype.bits();
        if ((prim_ty->dtype.is_int() && other_node->dtype.is_int()) ||
            (prim_ty->dtype.is_float() && other_node->dtype.is_float())) {
          return PrimType(runtime::DataType(prim_ty->dtype.code(), bits, prim_ty->dtype.lanes()));
        }
      }
    }
    return any_type;
  };
  if (auto* prim_ty = t1.as<PrimTypeNode>()) {
    return InferPrimType(prim_ty, t2);
  }
  if (auto* prim_ty = t2.as<PrimTypeNode>()) {
    return InferPrimType(prim_ty, t1);
  }
  auto InferClassType = [&](const ClassTypeNode* cls, const Type& other) -> Type {
    if (auto* other_node = other.as<ClassTypeNode>()) {
      return cls == other_node ? other : any_type;
    } else if (IsUserDataType(other)) {
      return UserDataType();
    } else {
      return any_type;
    }
  };
  if (auto* cls_ty = t1.as<ClassTypeNode>()) {
    return InferClassType(cls_ty, t2);
  }
  if (auto* cls_ty = t2.as<ClassTypeNode>()) {
    return InferClassType(cls_ty, t1);
  }

  auto InferDictType = [&](const DictTypeNode* dict_ty, const Type& other) -> Type {
    if (auto* other_node = other.as<DictTypeNode>()) {
      return DictType(dict_ty->is_full_typed,
                      InferLiftType(dict_ty->key_type, other_node->key_type),
                      InferLiftType(dict_ty->value_type, other_node->value_type));
    } else {
      return any_type;
    }
  };
  if (auto* dict_ty = t1.as<DictTypeNode>()) {
    return InferDictType(dict_ty, t2);
  }
  if (auto* dict_ty = t2.as<DictTypeNode>()) {
    return InferDictType(dict_ty, t1);
  }

  auto InferSetType = [&](const SetTypeNode* set_ty, const Type& other) -> Type {
    if (auto* other_node = other.as<SetTypeNode>()) {
      return SetType(set_ty->is_full_typed,
                     InferLiftType(set_ty->item_type, other_node->item_type));
    } else {
      return any_type;
    }
  };
  if (auto* set_ty = t1.as<SetTypeNode>()) {
    return InferSetType(set_ty, t2);
  }
  if (auto* set_ty = t2.as<SetTypeNode>()) {
    return InferSetType(set_ty, t1);
  }

  auto InferListType = [&](const ListTypeNode* list_ty, const Type& other) -> Type {
    if (auto* other_node = other.as<ListTypeNode>()) {
      return ListType(list_ty->is_full_typed,
                      InferLiftType(list_ty->item_type, other_node->item_type));
    } else {
      return any_type;
    }
  };
  if (auto* list_ty = t1.as<ListTypeNode>()) {
    return InferListType(list_ty, t2);
  }
  if (auto* list_ty = t2.as<ListTypeNode>()) {
    return InferListType(list_ty, t1);
  }

  auto InferTupleType = [&](const TupleTypeNode* tup_ty, const Type& other) -> Type {
    if (auto* other_node = other.as<TupleTypeNode>()) {
      if (tup_ty->fields.empty() || other_node->fields.empty()) {
        // TODO: fix Generic Tuple[...]
        return any_type;
      }
      if (tup_ty->fields.size() != other_node->fields.size()) {
        return any_type;
      }
      runtime::Array<Type> tup_fields;
      for (size_t i = 0; i < tup_ty->fields.size(); ++i) {
        tup_fields.push_back(InferLiftType(tup_ty->fields[i], other_node->fields[i]));
      }
      return TupleType(std::move(tup_fields));
    } else {
      return any_type;
    }
  };
  if (auto* tup_ty = t1.as<TupleTypeNode>()) {
    return InferTupleType(tup_ty, t2);
  }
  if (auto* tup_ty = t2.as<TupleTypeNode>()) {
    return InferTupleType(tup_ty, t1);
  }

  if (t1 == t2) {
    return t1;
  }
  return any_type;
}

MATXSCRIPT_REGISTER_GLOBAL("ir.InferIteratorValueType").set_body_typed([](Type value) {
  return InferIteratorValueType(value);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.InferNthItemType").set_body_typed([](Type value, int64_t index) {
  return InferNthItemType(value, index);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.IsTypeConvertible").set_body_typed([](Type from, Type to) {
  return IsTypeConvertible(from, to);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.InferLiftType").set_body_typed([](Type t1, Type t2) {
  return InferLiftType(t1, t2);
});

}  // namespace ir
}  // namespace matxscript
