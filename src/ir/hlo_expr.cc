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
#include <matxscript/ir/hlo_expr.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/adt.h>
#include <matxscript/ir/hlo_builtin.h>
#include <matxscript/ir/op_attr_types.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;
using namespace ::matxscript::ir::printer;

StringImm::StringImm(StringRef value, Span span) {
  ObjectPtr<StringImmNode> node = runtime::make_object<StringImmNode>();
  node->value = std::move(value);
  node->checked_type_ = StringType(true);
  node->span = std::move(span);
  data_ = std::move(node);
}
MATXSCRIPT_REGISTER_NODE_TYPE(StringImmNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StringImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StringImmNode*>(node.get());
      p->stream << op->value;
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.StringImm").set_body_typed([](StringRef s, Span span) {
  return StringImm(std::move(s), std::move(span));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<StringImm>("", [](StringImm s, ObjectPath p, IRDocsifier d) -> Doc {
      // TODO: fix bytes print
      return LiteralDoc::Str(s->value, p->Attr("value"));
    });

UnicodeImm::UnicodeImm(StringRef value, Span span) {
  ObjectPtr<UnicodeImmNode> node = runtime::make_object<UnicodeImmNode>();
  node->value = std::move(value);
  node->checked_type_ = UnicodeType(true);
  node->span = std::move(span);
  data_ = std::move(node);
}
MATXSCRIPT_REGISTER_NODE_TYPE(UnicodeImmNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UnicodeImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const UnicodeImmNode*>(node.get());
      p->stream << op->value;
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.UnicodeImm").set_body_typed([](StringRef s, Span span) {
  return UnicodeImm(std::move(s), std::move(span));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<UnicodeImm>("", [](UnicodeImm s, ObjectPath p, IRDocsifier d) -> Doc {
      // TODO: fix unicode
      return LiteralDoc::Str(s->value, p->Attr("value"));
    });

#define MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(Name)                       \
  Name::Name(BaseExpr a, BaseExpr b, Span span) {                       \
    using T = Name::ContainerType;                                      \
    MXCHECK(a.defined()) << "ValueError: a is undefined\n";             \
    MXCHECK(b.defined()) << "ValueError: b is undefined\n";             \
    MXCHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n"; \
    ObjectPtr<T> node = make_object<T>();                               \
    node->dtype = DataType::Bool(a.dtype().lanes());                    \
    node->checked_type_ = PrimType(node->dtype);                        \
    node->a = std::move(a);                                             \
    node->b = std::move(b);                                             \
    node->span = std::move(span);                                       \
    data_ = std::move(node);                                            \
  }

static Type InferAddOpType(const Type& lhs_raw, const Type& rhs_raw) {
  const auto& lhs_type = RemoveReference(lhs_raw);
  const auto& rhs_type = RemoveReference(rhs_raw);
  // TODO: check class methods: __add__ __radd__ __iadd__
  if (IsStringType(lhs_type)) {
    if (!(IsStringType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: can't concat '" << rhs_type->GetPythonTypeName() << "' to bytes";
    }
    return StringType();
  } else if (IsStringType(rhs_type)) {
    if (!(IsStringType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and 'bytes'";
    }
    return StringType();
  } else if (IsUnicodeType(lhs_type)) {
    if (!(IsUnicodeType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: can only concatenate str (not \"" << rhs_type->GetPythonTypeName()
              << "\") to str";
    }
    return UnicodeType();
  } else if (IsUnicodeType(rhs_type)) {
    if (!(IsUnicodeType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and 'str'";
    }
    return UnicodeType();
  } else if (auto* lhs_list_node = lhs_type.as<ListTypeNode>()) {
    if (auto* rhs_list_node = rhs_type.as<ListTypeNode>()) {
      if (lhs_list_node->item_type == rhs_list_node->item_type) {
        return ListType(lhs_list_node->IsFullTyped() && rhs_list_node->IsFullTyped(),
                        lhs_list_node->item_type);
      } else {
        return ListType();
      }
    } else if (IsObjectType(rhs_type)) {
      return ListType();
    } else {
      MXTHROW << "TypeError: can only concatenate list (not \"" << rhs_type->GetPythonTypeName()
              << "\") to list";
      return ListType();
    }
  } else if (auto* rhs_list_node = rhs_type.as<ListTypeNode>()) {
    if (auto* lhs_list_node = lhs_type.as<ListTypeNode>()) {
      if (lhs_list_node->item_type == rhs_list_node->item_type) {
        return ListType(lhs_list_node->IsFullTyped() && rhs_list_node->IsFullTyped(),
                        lhs_list_node->item_type);
      } else {
        return ListType();
      }
    } else if (IsObjectType(lhs_type)) {
      return ListType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and 'list'";
      return ListType();
    }
  } else if (auto* lhs_tup_node = lhs_type.as<TupleTypeNode>()) {
    if (auto* rhs_tup_node = rhs_type.as<TupleTypeNode>()) {
      Array<Type> fields = Concat(lhs_tup_node->fields, rhs_tup_node->fields);
      return TupleType(fields);
    } else if (IsObjectType(rhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: can only concatenate tuple (not \"" << rhs_type->GetPythonTypeName()
              << "\") to tuple";
      return ObjectType();
    }
  } else if (auto* rhs_tup_node = rhs_type.as<TupleTypeNode>()) {
    if (auto* lhs_tup_node = lhs_type.as<TupleTypeNode>()) {
      Array<Type> fields = Concat(lhs_tup_node->fields, rhs_tup_node->fields);
      return TupleType(fields);
    } else if (IsObjectType(lhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and 'tuple'";
      return ObjectType();
    }
  } else if (IsFloatType(lhs_type)) {
    if (!(IsPrimType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for +: 'float' and '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsFloatType(rhs_type)) {
    if (!(IsPrimType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and 'float'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsPrimType(lhs_type)) {
    if (IsPrimType(rhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(rhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else if (IsPrimType(rhs_type)) {
    if (IsPrimType(lhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(lhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else {
    if (!(IsObjectType(lhs_type) && IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for +: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
    }
  }
  return ObjectType();
}

#define MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(NodeType, OpKind)                                \
  MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                               \
      .set_dispatch<ir::NodeType>("", [](ir::NodeType node, ObjectPath p, IRDocsifier d) -> Doc { \
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));                                     \
        ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));                                     \
        return OperationDoc(OperationDocNode::Kind::OpKind, {a, b});                              \
      });

// HLOAdd
HLOAdd::HLOAdd(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  MXCHECK(!(IsPrimType(RemoveReference(a->checked_type_)) &&
            IsPrimType(RemoveReference(b->checked_type_))))
      << "TypeError: should use PrimAdd";
  ObjectPtr<HLOAddNode> node = make_object<HLOAddNode>();
  node->checked_type_ = InferAddOpType(a->checked_type_, b->checked_type_);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOAdd").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOAdd(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOAddNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOAddNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOAddNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " + ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOAdd, kAdd);

// HLOSub
static Type InferSubOpType(const Type& lhs_raw, const Type& rhs_raw) {
  const auto& lhs_type = RemoveReference(lhs_raw);
  const auto& rhs_type = RemoveReference(rhs_raw);
  // TODO: check class methods: __sub__ __rsub__ __isub__
  if (lhs_type.as<SetTypeNode>()) {
    if (IsSetType(rhs_type) || IsObjectType(rhs_type)) {
      return lhs_type;
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for -: 'set' and '"
              << rhs_type->GetPythonTypeName() << "'";
      return lhs_type;
    }
  } else if (rhs_type.as<SetTypeNode>()) {
    if (IsObjectType(lhs_type)) {
      // only support set - set
      return SetType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for -: '" << lhs_type->GetPythonTypeName()
              << "' and 'set'";
      return SetType();
    }
  } else if (IsFloatType(lhs_type)) {
    if (!(IsPrimType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for -: 'float' and '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsFloatType(rhs_type)) {
    if (!(IsPrimType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for -: '" << lhs_type->GetPythonTypeName()
              << "' and 'float'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsPrimType(lhs_type)) {
    if (IsPrimType(rhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(rhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for -: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else if (IsPrimType(rhs_type)) {
    if (IsPrimType(lhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(lhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for -: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else {
    if (!(IsObjectType(lhs_type) && IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for -: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
    }
  }
  return ObjectType();
}

HLOSub::HLOSub(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  MXCHECK(!(IsPrimType(RemoveReference(a->checked_type_)) &&
            IsPrimType(RemoveReference(b->checked_type_))))
      << "TypeError: should use PrimSub";
  ObjectPtr<HLOSubNode> node = make_object<HLOSubNode>();
  node->checked_type_ = InferSubOpType(a->checked_type_, b->checked_type_);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOSub").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOSub(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOSubNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOSubNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOSubNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " - ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOSub, kSub);

// HLOMul
static Type InferMulOpType(const Type& lhs_raw, const Type& rhs_raw) {
  const auto& lhs_type = RemoveReference(lhs_raw);
  const auto& rhs_type = RemoveReference(rhs_raw);
  // TODO: check class methods: __mul__ __rmul__ __imul__
  if (IsStringType(lhs_type)) {
    if (!(IsIntegerType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return StringType();
  } else if (IsStringType(rhs_type)) {
    if (!(IsIntegerType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << lhs_type->GetPythonTypeName() << "'";
    }
    return StringType();
  } else if (IsUnicodeType(lhs_type)) {
    if (!(IsIntegerType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return UnicodeType();
  } else if (IsUnicodeType(rhs_type)) {
    if (!(IsIntegerType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << lhs_type->GetPythonTypeName() << "'";
    }
    return UnicodeType();
  } else if (lhs_type.as<ListTypeNode>()) {
    if (!(IsIntegerType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return lhs_type;
  } else if (rhs_type.as<ListTypeNode>()) {
    if (!(IsIntegerType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << lhs_type->GetPythonTypeName() << "'";
    }
    return rhs_type;
  } else if (auto* lhs_tup_node = lhs_type.as<TupleTypeNode>()) {
    if (!(IsIntegerType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return ObjectType();
  } else if (auto* rhs_tup_node = rhs_type.as<TupleTypeNode>()) {
    if (!(IsIntegerType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: can't multiply sequence by non-int of type '"
              << lhs_type->GetPythonTypeName() << "'";
    }
    return ObjectType();
  } else if (IsFloatType(lhs_type)) {
    if (!(IsPrimType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for *: 'float' and '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsFloatType(rhs_type)) {
    if (!(IsPrimType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for *: '" << lhs_type->GetPythonTypeName()
              << "' and 'float'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsPrimType(lhs_type)) {
    if (IsPrimType(rhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(rhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for *: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else if (IsPrimType(rhs_type)) {
    if (IsPrimType(lhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(lhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for *: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else {
    if (!(IsObjectType(lhs_type) && IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for *: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
    }
  }
  return ObjectType();
}
HLOMul::HLOMul(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  MXCHECK(!(IsPrimType(RemoveReference(a->checked_type_)) &&
            IsPrimType(RemoveReference(b->checked_type_))))
      << "TypeError: should use PrimMul";
  ObjectPtr<HLOMulNode> node = make_object<HLOMulNode>();
  node->checked_type_ = InferMulOpType(a->checked_type_, b->checked_type_);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}
MATXSCRIPT_REGISTER_GLOBAL("ir.HLOMul").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOMul(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOMulNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOMulNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOMulNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << "*";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOMul, kMult);

// HLOFloorDiv
static Type InferFloorDivOpType(const Type& lhs_raw, const Type& rhs_raw) {
  const auto& lhs_type = RemoveReference(lhs_raw);
  const auto& rhs_type = RemoveReference(rhs_raw);
  // TODO: check class methods: __floordiv__ __rfloordiv__ __ifloordiv__
  if (IsFloatType(lhs_type)) {
    if (!(IsPrimType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for -: 'float' and '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsFloatType(rhs_type)) {
    if (!(IsPrimType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and 'float'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsPrimType(lhs_type)) {
    if (IsPrimType(rhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(rhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else if (IsPrimType(rhs_type)) {
    if (IsPrimType(lhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(lhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else {
    if (!(IsObjectType(lhs_type) && IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
    }
  }
  return ObjectType();
}

HLOFloorDiv::HLOFloorDiv(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLOFloorDivNode> node = make_object<HLOFloorDivNode>();
  node->checked_type_ = InferFloorDivOpType(a->checked_type_, b->checked_type_);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOFloorDiv").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOFloorDiv(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOFloorDivNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOFloorDivNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOFloorDivNode*>(node.get());
      p->stream << "floordiv(" << op->a << ", " << op->b << ")";
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOFloorDiv, kFloorDiv);

// HLOFloorMod
static Type InferFloorModOpType(const Type& lhs_raw, const Type& rhs_raw) {
  const auto& lhs_type = RemoveReference(lhs_raw);
  const auto& rhs_type = RemoveReference(rhs_raw);
  // TODO: check class methods: __mod__ __rmod__ __imod__
  if (IsFloatType(lhs_type)) {
    if (!(IsPrimType(rhs_type) || IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for -: 'float' and '"
              << rhs_type->GetPythonTypeName() << "'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsFloatType(rhs_type)) {
    if (!(IsPrimType(lhs_type) || IsObjectType(lhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and 'float'";
    }
    return PrimType(runtime::DataType::Float(64));
  } else if (IsPrimType(lhs_type)) {
    if (IsPrimType(rhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(rhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else if (IsPrimType(rhs_type)) {
    if (IsPrimType(lhs_type)) {
      return PrimType(runtime::DataType::Int(64));
    } else if (IsObjectType(lhs_type)) {
      return ObjectType();
    } else {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
      return ObjectType();
    }
  } else {
    if (!(IsObjectType(lhs_type) && IsObjectType(rhs_type))) {
      MXTHROW << "TypeError: unsupported operand type(s) for //: '" << lhs_type->GetPythonTypeName()
              << "' and '" << rhs_type->GetPythonTypeName() << "'";
    }
  }
  return ObjectType();
}

HLOFloorMod::HLOFloorMod(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLOFloorModNode> node = make_object<HLOFloorModNode>();
  node->checked_type_ = InferFloorModOpType(a->checked_type_, b->checked_type_);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOFloorMod").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOFloorMod(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOFloorModNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOFloorModNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOFloorModNode*>(node.get());
      p->stream << "floormod(" << op->a << ", " << op->b << ")";
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOFloorMod, kMod);

// HLOEqual
HLOEqual::HLOEqual(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLOEqualNode> node = make_object<HLOEqualNode>();
  node->checked_type_ = BoolType();
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOEqual").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOEqual(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOEqualNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOEqualNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOEqualNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " == ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOEqual, kEq);

// HLONotEqual
HLONotEqual::HLONotEqual(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLONotEqualNode> node = make_object<HLONotEqualNode>();
  node->checked_type_ = BoolType();
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLONotEqual").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLONotEqual(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLONotEqualNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLONotEqualNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLONotEqualNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " != ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLONotEqual, kNotEq);

// HLOLessThan
HLOLessThan::HLOLessThan(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLOLessThanNode> node = make_object<HLOLessThanNode>();
  node->checked_type_ = BoolType();
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOLessThan").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOLessThan(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOLessThanNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOLessThanNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOLessThanNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " < ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOLessThan, kLt);

// HLOLessEqual
HLOLessEqual::HLOLessEqual(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLOLessEqualNode> node = make_object<HLOLessEqualNode>();
  node->checked_type_ = BoolType();
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOLessEqual").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOLessEqual(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOLessEqualNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOLessEqualNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOLessEqualNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " <= ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOLessEqual, kLtE);

// HLOGreaterThan
HLOGreaterThan::HLOGreaterThan(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLOGreaterThanNode> node = make_object<HLOGreaterThanNode>();
  node->checked_type_ = BoolType();
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOGreaterThan")
    .set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
      return HLOGreaterThan(std::move(a), std::move(b), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(HLOGreaterThanNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOGreaterThanNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOGreaterThanNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " > ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOGreaterThan, kGt);

// HLOGreaterEqual
HLOGreaterEqual::HLOGreaterEqual(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined\n";
  MXCHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<HLOGreaterEqualNode> node = make_object<HLOGreaterEqualNode>();
  node->checked_type_ = BoolType();
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOGreaterEqual")
    .set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
      return HLOGreaterEqual(std::move(a), std::move(b), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(HLOGreaterEqualNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOGreaterEqualNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOGreaterEqualNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " >= ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOGreaterEqual, kGtE);

// HLOAnd
HLOAnd::HLOAnd(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined";
  MXCHECK(b.defined()) << "ValueError: b is undefined";
  const auto& a_type = RemoveReference(a->checked_type_);
  const auto& b_type = RemoveReference(b->checked_type_);
  ObjectPtr<HLOAndNode> node = make_object<HLOAndNode>();
  if (IsUnicodeType(a_type) && IsUnicodeType(b_type)) {
    node->checked_type_ = UnicodeType(false);
  } else if (IsStringType(a_type) && IsStringType(b_type)) {
    node->checked_type_ = StringType(false);
  } else if (IsObjectType(a_type) && IsObjectType(b_type)) {
    node->checked_type_ = ObjectType(false);
  } else if (a->checked_type_ == b->checked_type_) {
    node->checked_type_ = a->checked_type_;
  } else {
    node->checked_type_ = ObjectType();
  }
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOAnd").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOAnd(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOAndNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOAndNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOAndNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " and ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOAnd, kAnd);

// HLOOr
HLOOr::HLOOr(BaseExpr a, BaseExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined";
  MXCHECK(b.defined()) << "ValueError: b is undefined";

  ObjectPtr<HLOOrNode> node = make_object<HLOOrNode>();
  if (a->checked_type_ == b->checked_type_) {
    node->checked_type_ = a->checked_type_;
  } else {
    node->checked_type_ = ObjectType();
  }
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOOr").set_body_typed([](BaseExpr a, BaseExpr b, Span span) {
  return HLOOr(std::move(a), std::move(b), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOOrNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOOrNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOOrNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " or ";
      p->Print(op->b);
      p->stream << ')';
    });

MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY(HLOOr, kOr);

#undef MATXSCRIPT_SCRIPT_PRINTER_DEF_HLO_BINARY

// HLONot
HLONot::HLONot(BaseExpr a, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined";

  ObjectPtr<HLONotNode> node = make_object<HLONotNode>();
  node->checked_type_ = BoolType();
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLONot").set_body_typed([](BaseExpr a, Span span) {
  return HLONot(std::move(a), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLONotNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLONotNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLONotNode*>(node.get());
      p->stream << "(not ";
      p->Print(op->a);
      p->stream << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::HLONot>("", [](ir::HLONot node, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      return OperationDoc(OperationDocNode::Kind::kNot, {a});
    });

// Call
Call::Call(Type ret_type, HLOExpr op, Array<BaseExpr> args, Span span, Array<ObjectRef> type_args) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->checked_type_ = std::move(ret_type);
  n->op = std::move(op);
  n->args = std::move(args);
  n->type_args = std::move(type_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(CallNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.Call").set_body_typed([](Type ret_type,
                                                        HLOExpr op,
                                                        Array<BaseExpr> args,
                                                        Span span,
                                                        Array<ObjectRef> type_args) {
  return Call(
      std::move(ret_type), std::move(op), std::move(args), std::move(span), std::move(type_args));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CallNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CallNode*>(ref.get());
      p->stream << "CallNode(" << node->op << ", " << node->args << ", " << node->type_args
                << ") -> " << node->checked_type_;
    });

// reformat some ops

template <typename DocType, typename AST>
static inline Array<DocType> build_arrays(const Array<AST>& ast_list,
                                          ObjectPath p,
                                          IRDocsifier d,
                                          int start_pos) {
  Array<DocType> results;
  int n_args = ast_list.size();
  results.reserve(n_args);
  for (int i = start_pos; i < n_args; ++i) {
    results.push_back(d->AsDoc<DocType>(ast_list[i], p->ArrayIndex(i)));
  }
  return results;
};

static Doc BuiltinsPrintToDoc(ir::Call call, ObjectPath p, IRDocsifier d) {
  MXCHECK(call->args.size() >= 3) << "internal error";
  Array<StringRef> kw_keys;
  Array<ExprDoc> kw_values;
  bool fill_sep = true;
  if (const auto* sep_node = call->args[0].as<StringImmNode>()) {
    fill_sep = sep_node->value.view() != " ";
  }
  if (fill_sep) {
    auto sep = d->AsDoc<ExprDoc>(call->args[0], p->Attr("args")->ArrayIndex(0));
    kw_keys.push_back("sep");
    kw_values.push_back(sep);
  }
  bool fill_end = true;
  if (const auto* end_node = call->args[1].as<StringImmNode>()) {
    fill_end = end_node->value.view() != "\n";
  }
  if (fill_end) {
    auto end = d->AsDoc<ExprDoc>(call->args[1], p->Attr("args")->ArrayIndex(0));
    kw_keys.push_back("end");
    kw_values.push_back(end);
  }
  // ignore file
  int arg_pos = 3;
  Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, p, d, arg_pos);
  return IdDoc("print")->Call(args, kw_keys, kw_values);
}

static Doc CallFunctionToDoc(StringRef fn_name, ir::Call call, ObjectPath p, IRDocsifier d) {
  Array<StringRef> kw_keys;
  Array<ExprDoc> kw_values;
  runtime::string_view builtins("builtins.");
  if (runtime::StringHelper::StartsWith(fn_name, builtins)) {
    fn_name = fn_name.view().substr(builtins.size());
  }
  Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, p, d, 0);
  return IdDoc(fn_name)->Call(args, kw_keys, kw_values);
}

static Doc CallMethodToDoc(StringRef method_name, ir::Call call, ObjectPath p, IRDocsifier d) {
  Array<StringRef> kw_keys;
  Array<ExprDoc> kw_values;
  MXCHECK(call->args.size() >= 1) << "internal error";
  auto self = d->AsDoc<ExprDoc>(call->args[0], p->Attr("args")->ArrayIndex(0));
  int arg_pos = 1;
  if (method_name.view() == "__getitem__") {
    Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, p, d, arg_pos);
    return self[Downcast<Array<Doc>>(args)];
  } else if (method_name.view() == "__getslice__") {
    MXCHECK(call->args.size() == 4) << "internal error";
    Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, p, d, arg_pos);
    return self[{SliceDoc(args[0], args[1], args[2])}];
  } else if (method_name.view() == "__setitem__") {
    MXCHECK(call->args.size() == 3) << "internal error";
    auto lhs = d->AsDoc<Doc>(call->args[1], p->Attr("args")->ArrayIndex(1));
    auto rhs = d->AsDoc<ExprDoc>(call->args[2], p->Attr("args")->ArrayIndex(2));
    return AssignDoc(self[{lhs}], rhs, NullOpt);
  } else if (method_name.view() == "__len__") {
    MXCHECK(call->args.size() == 1) << "internal error";
    return IdDoc("len")->Call({self});
  } else if (method_name.view() == "__contains__") {
    MXCHECK(call->args.size() == 2) << "internal error";
    auto lhs = d->AsDoc<ExprDoc>(call->args[1], p->Attr("args")->ArrayIndex(1));
    return OperationDoc(OperationDocNode::Kind::kIn, {lhs, self});
  }
  Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, p, d, arg_pos);
  return self->Attr(method_name)->Call(args, kw_keys, kw_values);
}

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::Call>("", [](ir::Call call, ObjectPath call_p, IRDocsifier d) -> Doc {
      ExprDoc prefix{nullptr};
      Array<StringRef> kw_keys;
      Array<ExprDoc> kw_values;
      if (const auto* op = call->op.as<OpNode>()) {
        static OpAttrMap<TPrinterGlobalSymbol> op_global_symbol =
            Op::GetAttrMap<TPrinterGlobalSymbol>("TPrinterGlobalSymbol");
        static OpAttrMap<TPrinterMethodSymbol> op_method_symbol =
            Op::GetAttrMap<TPrinterMethodSymbol>("TPrinterMethodSymbol");

        auto op_ref = GetRef<Op>(op);
        if (op_ref.same_as(builtin::call_lambda())) {
          MXCHECK(call->args.size() == 1);
          return d->AsDoc<ExprDoc>(call->args[0], call_p->Attr("args")->ArrayIndex(0));
        } else if (op_ref.same_as(builtin::builtins_print())) {
          return BuiltinsPrintToDoc(call, call_p, d);
        } else if (op_global_symbol.count(op_ref)) {
          StringRef name = op_global_symbol[op_ref];
          return CallFunctionToDoc(name, call, call_p, d);
        } else if (op_method_symbol.count(op_ref)) {
          StringRef name = op_method_symbol[op_ref];
          return CallMethodToDoc(name, call, call_p, d);
        } else {
          StringRef name = op->name;
          prefix = Dialect(d, name);
        }
      } else if (const auto* gv = call->op.as<GlobalVarNode>()) {
        prefix = LiteralDoc::Str(gv->name_hint, call_p->Attr("op"));
      } else {
        prefix = d->AsDoc<ExprDoc>(call->op, call_p->Attr("op"));
      }
      Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, call_p, d, 0);
      return prefix->Call(args, kw_keys, kw_values);
    });

// HLOIterator
HLOIterator::HLOIterator(BaseExpr container, IntImm method, Span span) {
  ObjectPtr<HLOIteratorNode> n = make_object<HLOIteratorNode>();
  n->container = std::move(container);
  n->method = std::move(method);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(HLOIteratorNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOIterator")
    .set_body_typed([](BaseExpr container, int64_t method, Span span) {
      return HLOIterator(
          std::move(container), IntImm(runtime::DataType::Int(64), method), std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOIteratorNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const HLOIteratorNode*>(ref.get());
      p->stream << "HLOIterator(" << node->container << "." << node->method << ")";
    });

// TODO: remove unused HLOIterator

// InitializerList
InitializerList::InitializerList(Array<BaseExpr> fields, Span span) {
  ObjectPtr<InitializerListNode> n = make_object<InitializerListNode>();
  Type item_type = ObjectType();
  if (!fields.empty()) {
    bool is_same = true;
    auto& f_ty_0 = fields[0]->checked_type();
    if (f_ty_0.defined()) {
      for (auto i = 1; i < fields.size(); ++i) {
        auto& f_ty_i = fields[i]->checked_type();
        if ((!f_ty_i.defined()) ||
            f_ty_0->type_index() != fields[i]->checked_type()->type_index()) {
          is_same = false;
        }
      }
    }
    if (is_same) {
      item_type = f_ty_0;
    }
  }
  n->checked_type_ = ListType(false, std::move(item_type));
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(InitializerListNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.InitializerList")
    .set_body_typed([](Array<BaseExpr> fields, Span span) {
      return InitializerList(std::move(fields), std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<InitializerListNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const InitializerListNode*>(ref.get());
      p->stream << "InitializerList(" << node->fields << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::InitializerList>(
        "", [](ir::InitializerList li, ObjectPath li_p, IRDocsifier d) -> Doc {
          return d->AsDoc<ExprDoc>(li->fields, li_p->Attr("fields"));
        });

// InitializerDict
InitializerDict::InitializerDict(Map<BaseExpr, BaseExpr> fields, Span span) {
  ObjectPtr<InitializerDictNode> n = make_object<InitializerDictNode>();
  Type key_type = ObjectType();
  Type mapped_type = ObjectType();
  if (!fields.empty()) {
    bool key_is_same = true;
    auto iter = fields.begin();
    auto& key_ty_0 = (*iter).first->checked_type();
    iter++;
    if (key_ty_0.defined()) {
      for (; iter != fields.end(); ++iter) {
        auto& key_ty_i = (*iter).first->checked_type();
        auto& mapped_ty_i = (*iter).second->checked_type();
        if ((!key_ty_i.defined()) || key_ty_0->type_index() != key_ty_i->type_index()) {
          key_is_same = false;
        }
      }
    }
    bool mapped_is_same = true;
    iter = fields.begin();
    auto& mapped_ty_0 = (*iter).second->checked_type();
    if (mapped_ty_0.defined()) {
      for (; iter != fields.end(); ++iter) {
        auto& mapped_ty_i = (*iter).second->checked_type();
        if ((!mapped_ty_i.defined()) || mapped_ty_0->type_index() != mapped_ty_i->type_index()) {
          mapped_is_same = false;
        }
      }
    }
    if (key_is_same) {
      key_type = key_ty_0;
    }
    if (mapped_is_same) {
      mapped_type = mapped_ty_0;
    }
  }
  n->checked_type_ = DictType(false, std::move(key_type), std::move(mapped_type));
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(InitializerDictNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.InitializerDict")
    .set_body_typed([](Map<BaseExpr, BaseExpr> fields, Span span) {
      return InitializerDict(std::move(fields), std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<InitializerDictNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const InitializerDictNode*>(ref.get());
      p->stream << "InitializerDict(" << node->fields << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::InitializerDict>(
        "", [](ir::InitializerDict di, ObjectPath di_p, IRDocsifier d) -> Doc {
          return d->AsDoc<ExprDoc>(di->fields, di_p->Attr("fields"));
        });

// EnumAttr
EnumAttr::EnumAttr(StringRef enum_str, Span span) {
  ObjectPtr<EnumAttrNode> n = make_object<EnumAttrNode>();
  n->enum_str = std::move(enum_str);
  n->span = std::move(span);
  n->checked_type_ = PrimType(DataType::Int(64));
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(EnumAttrNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.EnumAttr").set_body_typed([](StringRef enum_str, Span span) {
  return EnumAttr(std::move(enum_str), std::move(span));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EnumAttrNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const EnumAttrNode*>(ref.get());
      p->stream << "EnumAttr(" << node->enum_str << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::EnumAttr>("", [](ir::EnumAttr en, ObjectPath en_p, IRDocsifier d) -> Doc {
      // TODO: fixme
      return d->AsDoc<ExprDoc>(en->enum_str, en_p->Attr("enum_str"));
    });

// ClassGetItem
ClassGetItem::ClassGetItem(HLOExpr self, StringImm attr, Span span) {
  ObjectPtr<ClassGetItemNode> n = make_object<ClassGetItemNode>();
  auto self_checked_ty = self->checked_type();
  if (IsRefType(self_checked_ty)) {
    self_checked_ty = Downcast<RefType>(self_checked_ty)->value;
  }
  ClassType self_ty = Downcast<ClassType>(self_checked_ty);
  n->checked_type_ = self_ty->GetItem(attr->value);
  n->self = std::move(self);
  n->attr = std::move(attr);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ClassGetItemNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassGetItem")
    .set_body_typed([](HLOExpr self, StringImm attr, Span span) {
      return ClassGetItem(std::move(self), std::move(attr), std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ClassGetItemNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ClassGetItemNode*>(ref.get());
      p->stream << "ClassGetItemNode(" << node->self << "." << node->attr << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::ClassGetItem>(
        "", [](ir::ClassGetItem cls_getitem, ObjectPath cls_getitem_p, IRDocsifier d) -> Doc {
          auto self = d->AsDoc<ExprDoc>(cls_getitem->self, cls_getitem_p->Attr("self"));
          // super()
          if (auto* self_ty = cls_getitem->self->checked_type_.as<ClassTypeNode>()) {
            if (self_ty->base.defined()) {
              if (auto* base_ty = self_ty->base.as<ClassTypeNode>()) {
                auto super_init_attr = base_ty->header->name_hint + "::__init__";
                if (cls_getitem->attr->value == super_init_attr) {
                  return IdDoc("super")->Call({})->Attr("__init__");
                }
              }
            }
          }
          return self->Attr(cls_getitem->attr->value);
        });

// HLOCast
HLOCast::HLOCast(Type t, BaseExpr value, Span span) {
  MXCHECK(value.defined());
  ObjectPtr<HLOCastNode> node = make_object<HLOCastNode>();
  node->checked_type_ = std::move(t);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOCast").set_body_typed([](Type ty, BaseExpr value, Span span) {
  return HLOCast(std::move(ty), std::move(value), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOCastNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOCastNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOCastNode*>(node.get());
      p->stream << op->checked_type_ << '(';
      p->Print(op->value);
      p->stream << ')';
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::HLOCast>("", [](ir::HLOCast e, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc value = d->AsDoc<ExprDoc>(e->value, p->Attr("value"));
      if (d->cfg->ignore_type_cast) {
        return value;
      }
      ExprDoc dtype = LiteralDoc::HLOType(e->checked_type_, p->Attr("checked_type_"));
      return Dialect(d, "HLOCast")->Call({dtype, value});
    });

// HLOMove
HLOMove::HLOMove(BaseExpr value, Span span) {
  MXCHECK(value.defined());
  ObjectPtr<HLOMoveNode> node = make_object<HLOMoveNode>();
  node->checked_type_ = value->checked_type_;
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOMove").set_body_typed([](BaseExpr value, Span span) {
  return HLOMove(std::move(value), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOMoveNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOMoveNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOMoveNode*>(node.get());
      p->stream << "move(";
      p->Print(op->value);
      p->stream << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::HLOMove>("", [](ir::HLOMove e, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc value = d->AsDoc<ExprDoc>(e->value, p->Attr("value"));
      return Dialect(d, "move")->Call({value});
    });

// HLOEnumerate
HLOEnumerate::HLOEnumerate(BaseExpr value, BaseExpr start, Span span) {
  MXCHECK(value.defined());
  ObjectPtr<HLOEnumerateNode> node = make_object<HLOEnumerateNode>();
  Type value_type = TupleType(
      {PrimType(runtime::DataType::Int(64)), InferIteratorValueType(value->checked_type())});
  node->checked_type_ =
      IteratorType(value->checked_type(), value_type, value->checked_type()->HasBeginEnd());
  node->value = std::move(value);
  if (start->IsInstance<PrimExprNode>()) {
    node->start = PrimCast(runtime::DataType::Int(64), Downcast<PrimExpr>(std::move(start)));
  } else {
    node->start = HLOCastPrim(runtime::DataType::Int(64), std::move(start));
  }
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOEnumerate")
    .set_body_typed([](BaseExpr value, BaseExpr start, Span span) {
      return HLOEnumerate(std::move(value), std::move(start), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(HLOEnumerateNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOEnumerateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOEnumerateNode*>(node.get());
      p->stream << "enumerate(";
      p->Print(op->value);
      p->stream << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::HLOEnumerate>("", [](ir::HLOEnumerate e, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc value = d->AsDoc<ExprDoc>(e->value, p->Attr("value"));
      return IdDoc("enumerate")->Call({value});
    });

// HLOZip
HLOZip::HLOZip(Array<BaseExpr> values, Span span) {
  MXCHECK(values.defined());
  ObjectPtr<HLOZipNode> node = make_object<HLOZipNode>();
  Array<Type> sub_value_types;
  bool has_begin_end = true;
  for (auto& cons : values) {
    sub_value_types.push_back(InferIteratorValueType(cons->checked_type()));
    if (!cons->checked_type()->HasBeginEnd()) {
      has_begin_end = false;
    }
  }
  Type value_type = TupleType(std::move(sub_value_types));
  node->checked_type_ = IteratorType(ObjectType(), value_type, has_begin_end);
  node->values = std::move(values);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOZip").set_body_typed([](Array<BaseExpr> value, Span span) {
  return HLOZip(std::move(value), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(HLOZipNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOZipNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOZipNode*>(node.get());
      p->stream << "zip(";
      p->Print(op->values);
      p->stream << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::HLOZip>("", [](ir::HLOZip e, ObjectPath p, IRDocsifier d) -> Doc {
      int n = e->values.size();
      Array<ExprDoc> results;
      results.reserve(n);
      p = p->Attr("values");
      for (int i = 0; i < n; ++i) {
        results.push_back(d->AsDoc<ExprDoc>(e->values[i], p->ArrayIndex(i)));
      }
      return IdDoc("zip")->Call(results);
    });

// Comprehension
Comprehension::Comprehension(BaseExpr target, BaseExpr iter, Array<BaseExpr> ifs) {
  ObjectPtr<ComprehensionNode> node = make_object<ComprehensionNode>();
  node->target = std::move(target);
  node->iter = std::move(iter);
  node->ifs = std::move(ifs);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.Comprehension")
    .set_body_typed([](BaseExpr target, BaseExpr iter, Array<BaseExpr> ifs) {
      return Comprehension(target, std::move(iter), std::move(ifs));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(ComprehensionNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::Comprehension>(
        "", [](ir::Comprehension e, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc target = d->AsDoc<ExprDoc>(e->target, p->Attr("target"));
          ExprDoc iter = d->AsDoc<ExprDoc>(e->iter, p->Attr("iter"));
          p = p->Attr("ifs");
          int n = e->ifs.size();
          Optional<Array<ExprDoc>> ifs_opt{nullptr};
          if (n > 0) {
            auto ifs = Array<ExprDoc>();
            ifs.reserve(n);
            for (int i = 0; i < n; ++i) {
              ifs.push_back(d->AsDoc<ExprDoc>(e->ifs[i], p->ArrayIndex(i)));
            }
            ifs_opt = ifs;
          }
          return ComprehensionDoc(target, iter, ifs_opt);
        });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComprehensionNode>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << IRTextPrinter::Print(node, NullOpt);
    });

// ListComp
ListComp::ListComp(Type ann_typed, BaseExpr elt, Array<Comprehension> generators, Span span) {
  ObjectPtr<ListCompNode> node = make_object<ListCompNode>();
  node->elt = std::move(elt);
  node->generators = std::move(generators);
  if (ann_typed.defined()) {
    node->checked_type_ = std::move(ann_typed);
  } else {
    node->checked_type_ = ListType(false, ObjectType(), span);
  }
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.ListComp")
    .set_body_typed([](Type ann_typed, BaseExpr elt, Array<Comprehension> generators, Span span) {
      return ListComp(std::move(ann_typed), std::move(elt), std::move(generators), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(ListCompNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::ListComp>("", [](ir::ListComp e, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc elt = d->AsDoc<ExprDoc>(e->elt, p->Attr("elt"));
      p = p->Attr("generators");
      int n = e->generators.size();
      Array<ComprehensionDoc> generators;
      generators.reserve(n);
      for (int i = 0; i < n; ++i) {
        generators.push_back(d->AsDoc<ComprehensionDoc>(e->generators[i], p->ArrayIndex(i)));
      }
      return ListCompDoc(elt, generators);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ListCompNode>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << IRTextPrinter::Print(node, NullOpt);
    });

// SetComp
SetComp::SetComp(Type ann_typed, BaseExpr elt, Array<Comprehension> generators, Span span) {
  ObjectPtr<SetCompNode> node = make_object<SetCompNode>();
  node->elt = std::move(elt);
  node->generators = std::move(generators);
  if (ann_typed.defined()) {
    node->checked_type_ = std::move(ann_typed);
  } else {
    node->checked_type_ = ListType(false, ObjectType(), span);
  }
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.SetComp")
    .set_body_typed([](Type ann_typed, BaseExpr elt, Array<Comprehension> generators, Span span) {
      return SetComp(std::move(ann_typed), std::move(elt), std::move(generators), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(SetCompNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::SetComp>("", [](ir::SetComp e, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc elt = d->AsDoc<ExprDoc>(e->elt, p->Attr("elt"));
      p = p->Attr("generators");
      int n = e->generators.size();
      Array<ComprehensionDoc> generators;
      generators.reserve(n);
      for (int i = 0; i < n; ++i) {
        generators.push_back(d->AsDoc<ComprehensionDoc>(e->generators[i], p->ArrayIndex(i)));
      }
      return SetCompDoc(elt, generators);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SetCompNode>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << IRTextPrinter::Print(node, NullOpt);
    });

// DictComp
DictComp::DictComp(
    Type ann_typed, BaseExpr key, BaseExpr value, Array<Comprehension> generators, Span span) {
  ObjectPtr<DictCompNode> node = make_object<DictCompNode>();
  node->key = std::move(key);
  node->value = std::move(value);
  node->generators = std::move(generators);
  if (ann_typed.defined()) {
    node->checked_type_ = std::move(ann_typed);
  } else {
    node->checked_type_ = ListType(false, ObjectType(), span);
  }
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.DictComp")
    .set_body_typed([](Type ann_typed,
                       BaseExpr key,
                       BaseExpr value,
                       Array<Comprehension> generators,
                       Span span) {
      return DictComp(std::move(ann_typed),
                      std::move(key),
                      std::move(value),
                      std::move(generators),
                      std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(DictCompNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::DictComp>("", [](ir::DictComp e, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc key = d->AsDoc<ExprDoc>(e->key, p->Attr("key"));
      ExprDoc value = d->AsDoc<ExprDoc>(e->value, p->Attr("value"));
      p = p->Attr("generators");
      int n = e->generators.size();
      Array<ComprehensionDoc> generators;
      generators.reserve(n);
      for (int i = 0; i < n; ++i) {
        generators.push_back(d->AsDoc<ComprehensionDoc>(e->generators[i], p->ArrayIndex(i)));
      }
      return DictCompDoc(key, value, generators);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DictCompNode>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << IRTextPrinter::Print(node, NullOpt);
    });

}  // namespace ir
}  // namespace matxscript
