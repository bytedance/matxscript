// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the ExprFunctor is inspired by TVM.
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
 * \file expr_functor.cc
 */
#include <matxscript/ir/expr_functor.h>
#include "functor_common.h"

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::Downcast;
using ::matxscript::runtime::GetRef;

/******************************************************************************
 * ExprVisitor
 *****************************************************************************/

#define DEFINE_BINOP_VISIT_(OP)                \
  void ExprVisitor::VisitExpr_(const OP* op) { \
    this->VisitExpr(op->a);                    \
    this->VisitExpr(op->b);                    \
  }

DEFINE_BINOP_VISIT_(PrimAddNode);
DEFINE_BINOP_VISIT_(PrimSubNode);
DEFINE_BINOP_VISIT_(PrimMulNode);
DEFINE_BINOP_VISIT_(PrimDivNode);
DEFINE_BINOP_VISIT_(PrimModNode);
DEFINE_BINOP_VISIT_(PrimFloorDivNode);
DEFINE_BINOP_VISIT_(PrimFloorModNode);
DEFINE_BINOP_VISIT_(PrimMinNode);
DEFINE_BINOP_VISIT_(PrimMaxNode);
DEFINE_BINOP_VISIT_(PrimEQNode);
DEFINE_BINOP_VISIT_(PrimNENode);
DEFINE_BINOP_VISIT_(PrimLTNode);
DEFINE_BINOP_VISIT_(PrimLENode);
DEFINE_BINOP_VISIT_(PrimGTNode);
DEFINE_BINOP_VISIT_(PrimGENode);
DEFINE_BINOP_VISIT_(PrimAndNode);
DEFINE_BINOP_VISIT_(PrimOrNode);

DEFINE_BINOP_VISIT_(HLOAddNode);
DEFINE_BINOP_VISIT_(HLOSubNode);
DEFINE_BINOP_VISIT_(HLOMulNode);
DEFINE_BINOP_VISIT_(HLOFloorDivNode);
DEFINE_BINOP_VISIT_(HLOFloorModNode);
DEFINE_BINOP_VISIT_(HLOEqualNode);
DEFINE_BINOP_VISIT_(HLONotEqualNode);
DEFINE_BINOP_VISIT_(HLOLessThanNode);
DEFINE_BINOP_VISIT_(HLOLessEqualNode);
DEFINE_BINOP_VISIT_(HLOGreaterThanNode);
DEFINE_BINOP_VISIT_(HLOGreaterEqualNode);
DEFINE_BINOP_VISIT_(HLOAndNode);
DEFINE_BINOP_VISIT_(HLOOrNode);

void ExprVisitor::VisitExpr_(const IntImmNode* op) {
}
void ExprVisitor::VisitExpr_(const FloatImmNode* op) {
}
void ExprVisitor::VisitExpr_(const StringImmNode* op) {
}
void ExprVisitor::VisitExpr_(const UnicodeImmNode* op) {
}

void ExprVisitor::VisitExpr_(const PrimNotNode* op) {
  this->VisitExpr(op->a);
}

void ExprVisitor::VisitExpr_(const HLONotNode* op) {
  this->VisitExpr(op->a);
}

void ExprVisitor::VisitExpr_(const PrimSelectNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->true_value);
  this->VisitExpr(op->false_value);
}

void ExprVisitor::VisitExpr_(const PrimVarNode* op) {
}

void ExprVisitor::VisitExpr_(const PrimCastNode* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const HLOCastPrimNode* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const PrimLetNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const PrimCallNode* op) {
  VisitArray(op->args, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const HLOVarNode* op) {
  this->VisitSpan(op->span);
  if (op->type_annotation.defined()) {
    this->VisitType(op->type_annotation);
  }
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const RangeExprNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->start);
  this->VisitExpr(op->stop);
  this->VisitExpr(op->step);
}

void ExprVisitor::VisitExpr_(const TupleExprNode* op) {
  this->VisitSpan(op->span);
  for (auto field : op->fields) {
    if (field->IsInstance<PrimExprNode>()) {
      this->VisitExpr(Downcast<PrimExpr>(field));
    } else {
      this->VisitExpr(Downcast<HLOExpr>(field));
    }
  }
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);

  for (auto ty_arg : op->type_args) {
    if (ty_arg->IsInstance<TypeNode>()) {
      this->VisitType(runtime::Downcast<Type>(ty_arg));
    } else {
      this->VisitExpr(runtime::Downcast<IntImm>(ty_arg));
    }
  }

  for (auto arg : op->args) {
    this->VisitExpr(arg);
  }
}

void ExprVisitor::VisitExpr_(const ConstructorNode* op) {
  this->VisitSpan(op->span);
  if (op->inputs.defined()) {
    for (const Type& t : op->inputs) {
      this->VisitType(t);
    }
  }
  this->VisitType(op->belong_to);
}

void ExprVisitor::VisitExpr_(const InitializerListNode* op) {
  this->VisitSpan(op->span);
  for (auto field : op->fields) {
    this->VisitExpr(field);
  }
}

void ExprVisitor::VisitExpr_(const InitializerDictNode* op) {
  this->VisitSpan(op->span);
  for (auto itr = op->fields.begin(); itr != op->fields.end(); ++itr) {
    this->VisitExpr((*itr).first);
    this->VisitExpr((*itr).second);
  }
}

void ExprVisitor::VisitExpr_(const HLOIteratorNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->container);
  this->VisitExpr(op->method);
}

void ExprVisitor::VisitExpr_(const OpNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const EnumAttrNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const ClassGetItemNode* op) {
  this->VisitExpr(op->self);
  this->VisitExpr(op->attr);
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const NoneExprNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const HLOCastNode* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const HLOMoveNode* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const HLOEnumerateNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->start);
}

void ExprVisitor::VisitExpr_(const HLOZipNode* op) {
  for (auto value : op->values) {
    this->VisitExpr(value);
  }
}

// other info
void ExprVisitor::VisitType(const Type& t) {
  return;
}

void ExprVisitor::VisitSpan(const Span& span) {
  return;
}

/******************************************************************************
 * ExprMutator
 *****************************************************************************/

#define DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(OP, EXPR) \
  EXPR ExprMutator::VisitExpr_(const OP* op) {       \
    return GetRef<EXPR>(op);                         \
  }

DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(IntImmNode, PrimExpr)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(FloatImmNode, PrimExpr)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(StringImmNode, HLOExpr)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(UnicodeImmNode, HLOExpr)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(PrimVarNode, PrimExpr)

#define DEFINE_BIOP_EXPR_MUTATE_(OP)                     \
  PrimExpr ExprMutator::VisitExpr_(const OP##Node* op) { \
    PrimExpr a = this->VisitExpr(op->a);                 \
    PrimExpr b = this->VisitExpr(op->b);                 \
    if (a.same_as(op->a) && b.same_as(op->b)) {          \
      return GetRef<PrimExpr>(op);                       \
    } else {                                             \
      return OP(std::move(a), std::move(b), op->span);   \
    }                                                    \
  }

DEFINE_BIOP_EXPR_MUTATE_(PrimAdd);
DEFINE_BIOP_EXPR_MUTATE_(PrimSub);
DEFINE_BIOP_EXPR_MUTATE_(PrimMul);
DEFINE_BIOP_EXPR_MUTATE_(PrimDiv);
DEFINE_BIOP_EXPR_MUTATE_(PrimMod);
DEFINE_BIOP_EXPR_MUTATE_(PrimFloorDiv);
DEFINE_BIOP_EXPR_MUTATE_(PrimFloorMod);
DEFINE_BIOP_EXPR_MUTATE_(PrimMin);
DEFINE_BIOP_EXPR_MUTATE_(PrimMax);
DEFINE_BIOP_EXPR_MUTATE_(PrimEQ);
DEFINE_BIOP_EXPR_MUTATE_(PrimNE);
DEFINE_BIOP_EXPR_MUTATE_(PrimLT);
DEFINE_BIOP_EXPR_MUTATE_(PrimLE);
DEFINE_BIOP_EXPR_MUTATE_(PrimGT);
DEFINE_BIOP_EXPR_MUTATE_(PrimGE);
DEFINE_BIOP_EXPR_MUTATE_(PrimAnd);
DEFINE_BIOP_EXPR_MUTATE_(PrimOr);

#define DEFINE_HLO_BIOP_EXPR_MUTATE_(OP)                \
  HLOExpr ExprMutator::VisitExpr_(const OP##Node* op) { \
    BaseExpr a = this->VisitExpr(op->a);                \
    BaseExpr b = this->VisitExpr(op->b);                \
    if (a.same_as(op->a) && b.same_as(op->b)) {         \
      return GetRef<HLOExpr>(op);                       \
    } else {                                            \
      return OP(std::move(a), std::move(b), op->span);  \
    }                                                   \
  }

DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOAdd);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOSub);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOMul);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOFloorDiv);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOFloorMod);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOEqual);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLONotEqual);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOLessThan);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOLessEqual);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOGreaterThan);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOGreaterEqual);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOAnd);
DEFINE_HLO_BIOP_EXPR_MUTATE_(HLOOr);

PrimExpr ExprMutator::VisitExpr_(const PrimNotNode* op) {
  PrimExpr a = this->VisitExpr(op->a);
  if (a.same_as(op->a)) {
    return GetRef<PrimExpr>(op);
  } else {
    return PrimNot(a, op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const HLONotNode* op) {
  BaseExpr a = this->VisitExpr(op->a);
  if (a.same_as(op->a)) {
    return GetRef<HLOExpr>(op);
  } else {
    return HLONot(a, op->span);
  }
}

PrimExpr ExprMutator::VisitExpr_(const PrimSelectNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr true_value = this->VisitExpr(op->true_value);
  PrimExpr false_value = this->VisitExpr(op->false_value);
  if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return PrimSelect(condition, true_value, false_value, op->span);
  }
}

PrimExpr ExprMutator::VisitExpr_(const PrimLetNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  PrimExpr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<PrimExpr>(op);
  } else {
    // TODO(matx4) : review other let bindding ?
    return PrimLet(op->var, value, body, op->span);
  }
}

PrimExpr ExprMutator::VisitExpr_(const PrimCallNode* op) {
  auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  runtime::Array<PrimExpr> args = MutateArray(op->args, fmutate);

  if (args.same_as(op->args)) {
    return GetRef<PrimExpr>(op);
  } else {
    return PrimCall(op->dtype, op->op, args, op->span);
  }
}

PrimExpr ExprMutator::VisitExpr_(const PrimCastNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return PrimCast(op->dtype, value, op->span);
  }
}

PrimExpr ExprMutator::VisitExpr_(const HLOCastPrimNode* op) {
  BaseExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<HLOCastPrim>(op);
  } else {
    return HLOCastPrim(op->dtype, value, op->span);
  }
}

// Begin HLO Expr
HLOExpr ExprMutator::VisitExpr_(const HLOVarNode* op) {
  if (op->type_annotation.defined()) {
    auto type = this->VisitType(op->type_annotation);
    if (!op->type_annotation.same_as(type)) {
      return HLOVar(op->vid, type, op->span);
    }
  }
  // default case return self.
  return GetRef<HLOExpr>(op);
}

HLOExpr ExprMutator::VisitExpr_(const GlobalVarNode* op) {
  return GetRef<HLOExpr>(op);
}

HLOExpr ExprMutator::VisitExpr_(const CallNode* call_node) {
  auto new_op = this->VisitExpr(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  runtime::Array<ObjectRef> ty_args;
  for (auto ty_arg : call_node->type_args) {
    if (ty_arg->IsInstance<TypeNode>()) {
      auto new_ty_arg = this->VisitType(runtime::Downcast<Type>(ty_arg));
      ty_args.push_back(new_ty_arg);
      unchanged &= new_ty_arg.same_as(ty_arg);
    } else {
      auto new_imm = this->VisitExpr(runtime::Downcast<IntImm>(ty_arg));
      ty_args.push_back(new_imm);
      unchanged &= new_imm.same_as(ty_arg);
    }
  }

  runtime::Array<BaseExpr> call_args;
  for (auto arg : call_node->args) {
    auto new_arg = this->VisitExpr(arg);
    call_args.push_back(new_arg);
    unchanged &= new_arg.same_as(arg);
  }

  if (unchanged) {
    return GetRef<HLOExpr>(call_node);
  } else {
    return Call(
        call_node->checked_type_, Downcast<HLOExpr>(new_op), call_args, call_node->span, ty_args);
  }
}

HLOExpr ExprMutator::VisitExpr_(const ConstructorNode* c) {
  return GetRef<HLOExpr>(c);
}

HLOExpr ExprMutator::VisitExpr_(const InitializerListNode* op) {
  runtime::Array<BaseExpr> fields;
  bool all_fields_unchanged = true;
  for (auto field : op->fields) {
    if (field->IsInstance<PrimExprNode>()) {
      auto new_field = this->VisitExpr(Downcast<PrimExpr>(field));
      fields.push_back(new_field);
      all_fields_unchanged &= new_field.same_as(field);
    } else {
      auto new_field = this->VisitExpr(Downcast<HLOExpr>(field));
      fields.push_back(new_field);
      all_fields_unchanged &= new_field.same_as(field);
    }
  }
  if (all_fields_unchanged) {
    return GetRef<HLOExpr>(op);
  } else {
    return InitializerList(fields, op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const InitializerDictNode* op) {
  runtime::Map<BaseExpr, BaseExpr> new_fields;
  runtime::Array<BaseExpr> values;
  bool all_fields_unchanged = true;
  for (auto itr = op->fields.begin(); itr != op->fields.end(); ++itr) {
    auto new_key = VisitExpr((*itr).first);
    auto new_value = VisitExpr((*itr).second);
    all_fields_unchanged &= new_key.same_as((*itr).first);
    all_fields_unchanged &= new_value.same_as((*itr).second);
    new_fields.Set(new_key, new_value);
  }
  if (all_fields_unchanged) {
    return GetRef<HLOExpr>(op);
  } else {
    return InitializerDict(new_fields, op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const HLOIteratorNode* op) {
  BaseExpr container = this->VisitExpr(op->container);
  BaseExpr method = this->VisitExpr(op->method);
  if (container.same_as(op->container) && method.same_as(op->method)) {
    return GetRef<HLOExpr>(op);
  } else {
    return HLOIterator(container, Downcast<IntImm>(method), op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const OpNode* op) {
  // op can not be changed currently
  return GetRef<HLOExpr>(op);
}

HLOExpr ExprMutator::VisitExpr_(const EnumAttrNode* op) {
  // EnumAttr can not be changed currently
  return GetRef<EnumAttr>(op);
}

HLOExpr ExprMutator::VisitExpr_(const ClassGetItemNode* op) {
  HLOExpr self = Downcast<HLOExpr>(this->VisitExpr(op->self));
  StringImm attr = Downcast<StringImm>(this->VisitExpr(op->attr));
  if (self.same_as(op->self) && attr.same_as(op->attr)) {
    return GetRef<HLOExpr>(op);
  } else {
    return ClassGetItem(self, attr, op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const NoneExprNode* op) {
  return GetRef<HLOExpr>(op);
}

HLOExpr ExprMutator::VisitExpr_(const HLOCastNode* op) {
  auto value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<HLOExpr>(op);
  } else {
    return HLOCast(op->checked_type_, value, op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const HLOMoveNode* op) {
  auto value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<HLOExpr>(op);
  } else {
    return HLOMove(std::move(value), op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const HLOEnumerateNode* op) {
  auto value = this->VisitExpr(op->value);
  auto start = this->VisitExpr(op->start);
  if (value.same_as(op->value) && start.same_as(op->start)) {
    return GetRef<HLOExpr>(op);
  } else {
    return HLOEnumerate(std::move(value), start, op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const HLOZipNode* op) {
  runtime::Array<BaseExpr> values;
  bool all_fields_unchanged = true;
  for (auto value : op->values) {
    auto new_value = this->VisitExpr(value);
    values.push_back(new_value);
    all_fields_unchanged &= value.same_as(new_value);
  }
  if (all_fields_unchanged) {
    return GetRef<HLOExpr>(op);
  } else {
    return HLOZip(std::move(values), op->span);
  }
}

Type ExprMutator::VisitType(const Type& t) {
  return t;
}

// kernel or script
HLOExpr ExprMutator::VisitExpr_(const TupleExprNode* op) {
  runtime::Array<BaseExpr> fields;
  bool all_fields_unchanged = true;
  for (auto field : op->fields) {
    if (field->IsInstance<PrimExprNode>()) {
      auto new_field = this->VisitExpr(Downcast<PrimExpr>(field));
      fields.push_back(new_field);
      all_fields_unchanged &= new_field.same_as(field);
    } else {
      auto new_field = this->VisitExpr(Downcast<HLOExpr>(field));
      fields.push_back(new_field);
      all_fields_unchanged &= new_field.same_as(field);
    }
  }
  if (all_fields_unchanged) {
    return GetRef<HLOExpr>(op);
  } else {
    return TupleExpr(fields, op->span);
  }
}

HLOExpr ExprMutator::VisitExpr_(const RangeExprNode* op) {
  auto start = this->VisitExpr(op->start);
  auto stop = this->VisitExpr(op->stop);
  auto step = this->VisitExpr(op->step);
  if (start.same_as(op->start) && stop.same_as(op->stop) && step.same_as(op->step)) {
    return GetRef<HLOExpr>(op);
  } else {
    return RangeExpr(std::move(start), std::move(stop), std::move(step), op->span);
  }
}

}  // namespace ir
}  // namespace matxscript
