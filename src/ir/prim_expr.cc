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
#include <matxscript/ir/prim_expr.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;

PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm(runtime::DataType::Int(32), value)) {
}

PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(runtime::DataType::Float(32), value)) {
}

IntImm::IntImm(runtime::DataType dtype, int64_t value, Span span) {
  MXCHECK(dtype.is_scalar()) << "ValueError: IntImm can only take scalar.";
  MXCHECK(dtype.is_int() || dtype.is_uint())
      << "ValueError: IntImm supports only int or uint type.";
  if (dtype.is_uint()) {
    MXCHECK_GE(value, 0U);
  }
  ObjectPtr<IntImmNode> node = runtime::make_object<IntImmNode>();
  node->dtype = dtype;
  node->checked_type_ = PrimType(node->dtype);
  node->value = value;
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(IntImmNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ir::IntImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ir::IntImmNode*>(node.get());
      if (op->dtype == runtime::DataType::Int(32)) {
        p->stream << op->value;
      } else {
        p->stream << "(" << op->dtype << ")" << op->value;
      }
    });

FloatImm::FloatImm(runtime::DataType dtype, double value, Span span) {
  MXCHECK_EQ(dtype.lanes(), 1) << "ValueError: FloatImm can only take scalar.";
  ObjectPtr<FloatImmNode> node = runtime::make_object<FloatImmNode>();
  node->dtype = dtype;
  node->checked_type_ = PrimType(node->dtype);
  node->value = value;
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(FloatImmNode);
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FloatImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FloatImmNode*>(node.get());
      auto& stream = p->stream;
      switch (op->dtype.bits()) {
        case 64:
          stream << op->value;
          break;
        case 32:
          stream << op->value << 'f';
          break;
        case 16:
          stream << op->value << 'h';
          break;
        default:
          MXLOG(FATAL) << "Unknown float type bits=" << op->dtype.bits();
      }
    });

// PrimCast
PrimCast::PrimCast(DataType t, PrimExpr value, Span span) {
  MXCHECK(value.defined());
  MXCHECK_EQ(t.lanes(), value.dtype().lanes());
  ObjectPtr<PrimCastNode> node = make_object<PrimCastNode>();
  node->dtype = t;
  node->checked_type_ = PrimType(node->dtype);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimCast")
    .set_body_typed([](DataType dtype, PrimExpr value, Span span) {
      return PrimCast(dtype, std::move(value), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(PrimCastNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimCastNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimCastNode*>(node.get());
      p->stream << op->dtype << '(';
      p->Print(op->value);
      p->stream << ')';
    });

// HLOCastPrim
HLOCastPrim::HLOCastPrim(DataType t, BaseExpr value, Span span) {
  MXCHECK(value.defined());
  ObjectPtr<HLOCastPrimNode> node = make_object<HLOCastPrimNode>();
  node->dtype = t;
  node->checked_type_ = PrimType(node->dtype);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOCastPrim")
    .set_body_typed([](DataType dtype, BaseExpr value, Span span) {
      return HLOCastPrim(dtype, value);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(HLOCastPrimNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOCastPrimNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HLOCastPrimNode*>(node.get());
      p->stream << op->dtype << '(';
      p->Print(op->value);
      p->stream << ')';
    });

#define MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(Name)                       \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                       \
    using T = Name::ContainerType;                                      \
    MXCHECK(a.defined()) << "ValueError: a is undefined\n";             \
    MXCHECK(b.defined()) << "ValueError: b is undefined\n";             \
    MXCHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n"; \
    ObjectPtr<T> node = make_object<T>();                               \
    node->dtype = a.dtype();                                            \
    node->checked_type_ = PrimType(node->dtype);                        \
    node->a = std::move(a);                                             \
    node->b = std::move(b);                                             \
    node->span = std::move(span);                                       \
    data_ = std::move(node);                                            \
  }

#define MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(Name)                       \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                       \
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

// PrimAdd
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimAdd);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimAdd").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimAdd(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimAddNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimAddNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimAddNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " + ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimSub
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimSub);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimSub").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimSub(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimSubNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimSubNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimSubNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " - ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimMul
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimMul);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimMul").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMul(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimMulNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimMulNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimMulNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << "*";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimDiv
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimDiv);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimDiv").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimDiv(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimDivNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimDivNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimDivNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << "/";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimMod
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimMod);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimMod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMod(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimModNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimModNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimModNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " % ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimFloorDiv
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimFloorDiv);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimFloorDiv").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimFloorDiv(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimFloorDivNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFloorDivNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimFloorDivNode*>(node.get());
      p->stream << "floordiv(" << op->a << ", " << op->b << ")";
    });

// PrimFloorMod
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimFloorMod);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimFloorMod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimFloorMod(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimFloorModNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFloorModNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimFloorModNode*>(node.get());
      p->stream << "floormod(" << op->a << ", " << op->b << ")";
    });

// PrimMin
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimMin);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimMin").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMin(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimMinNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimMinNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimMinNode*>(node.get());
      p->stream << "min(";
      p->Print(op->a);
      p->stream << ", ";
      p->Print(op->b);
      p->stream << ")";
    });

// PrimMax
MATXSCRIPT_DEFINE_BINOP_CONSTRUCTOR(PrimMax);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimMax").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMax(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimMaxNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimMaxNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimMaxNode*>(node.get());
      p->stream << "max(";
      p->Print(op->a);
      p->stream << ", ";
      p->Print(op->b);
      p->stream << ")";
    });

// PrimEQ
MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(PrimEQ);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimEQ").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimEQ(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimEQNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimEQNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimEQNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " == ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimNE
MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(PrimNE);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimNE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimNE(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimNENode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimNENode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimNENode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " != ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimLT
MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(PrimLT);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimLT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimLT(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimLTNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimLTNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimLTNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " < ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimLE
MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(PrimLE);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimLE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimLE(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimLENode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimLENode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimLENode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " <= ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimGT
MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(PrimGT);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimGT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimGT(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimGTNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimGTNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimGTNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " > ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimGE
MATXSCRIPT_DEFINE_CMPOP_CONSTRUCTOR(PrimGE);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimGE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimGE(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimGENode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimGENode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimGENode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " >= ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimAnd
PrimAnd::PrimAnd(PrimExpr a, PrimExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined";
  MXCHECK(b.defined()) << "ValueError: b is undefined";
  MXCHECK(a.dtype().is_bool() || a.dtype().is_int());
  MXCHECK(b.dtype().is_bool() || b.dtype().is_int());

  ObjectPtr<PrimAndNode> node = make_object<PrimAndNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimAnd").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimAnd(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimAndNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimAndNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimAndNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " && ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimOr
PrimOr::PrimOr(PrimExpr a, PrimExpr b, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined";
  MXCHECK(b.defined()) << "ValueError: b is undefined";
  MXCHECK(a.dtype().is_bool() || a.dtype().is_int());
  MXCHECK(b.dtype().is_bool() || b.dtype().is_int());

  ObjectPtr<PrimOrNode> node = make_object<PrimOrNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimOr").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimOr(std::move(a), std::move(b), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimOrNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimOrNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimOrNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " || ";
      p->Print(op->b);
      p->stream << ')';
    });

// PrimNot
PrimNot::PrimNot(PrimExpr a, Span span) {
  MXCHECK(a.defined()) << "ValueError: a is undefined";
  MXCHECK(a.dtype().is_bool() || a.dtype().is_int());

  ObjectPtr<PrimNotNode> node = make_object<PrimNotNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimNot").set_body_typed([](PrimExpr a, Span span) {
  return PrimNot(std::move(a), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(PrimNotNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimNotNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimNotNode*>(node.get());
      p->stream << '!';
      p->Print(op->a);
    });

// PrimSelect
PrimSelect::PrimSelect(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  MXCHECK(condition.defined()) << "ValueError: condition is undefined";
  MXCHECK(true_value.defined()) << "ValueError: true_value is undefined";
  MXCHECK(false_value.defined()) << "ValueError: true_value is undefined";
  MXCHECK(condition.dtype().is_bool());
  MXCHECK(condition.dtype().lanes() == true_value.dtype().lanes() ||
          condition.dtype().lanes() == 1);
  MXCHECK(false_value.dtype() == true_value.dtype()) << "TypeError: mismatched types";

  ObjectPtr<PrimSelectNode> node = make_object<PrimSelectNode>();
  node->dtype = true_value.dtype();
  node->checked_type_ = PrimType(node->dtype);
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimSelect")
    .set_body_typed([](PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
      return PrimSelect(std::move(condition), std::move(true_value), std::move(false_value), span);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(PrimSelectNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimSelectNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimSelectNode*>(node.get());
      p->stream << "select(";
      p->Print(op->condition);
      p->stream << ", ";
      p->Print(op->true_value);
      p->stream << ", ";
      p->Print(op->false_value);
      p->stream << ")";
    });

// Let
PrimLet::PrimLet(PrimVar var, PrimExpr value, PrimExpr body, Span span) {
  MXCHECK(value.defined());
  MXCHECK(body.defined());
  MXCHECK(var.as<PrimExprNode>());
  MXCHECK_EQ(value.dtype(), var.as<PrimExprNode>()->dtype);

  ObjectPtr<PrimLetNode> node = make_object<PrimLetNode>();
  node->dtype = body.dtype();
  node->checked_type_ = PrimType(node->dtype);
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.Let").set_body_typed(
    [](PrimVar var, PrimExpr value, PrimExpr body, Span span) {
      return PrimLet(var, value, body, span);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(PrimLetNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimLetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimLetNode*>(node.get());
      p->stream << "(let " << op->var << " = ";
      p->Print(op->value);
      p->stream << " in ";
      p->Print(op->body);
      p->stream << ")";
    });

// Call
PrimCall::PrimCall(DataType dtype, HLOExpr op, Array<PrimExpr> args, Span span) {
  for (size_t i = 0; i < args.size(); ++i) {
    MXCHECK(args[i].defined());
  }

  ObjectPtr<PrimCallNode> node = make_object<PrimCallNode>();
  node->dtype = dtype;
  node->checked_type_ = PrimType(node->dtype);
  node->op = std::move(op);
  node->args = std::move(args);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimCall")
    .set_body_typed([](DataType type, HLOExpr op, Array<ObjectRef> args, Span span) {
      Array<PrimExpr> prim_expr_args;
      for (const auto& it : args) {
        MXCHECK(it->IsInstance<PrimExprNode>());
        prim_expr_args.push_back(Downcast<PrimExpr>(it));
      }
      return PrimCall(type, op, prim_expr_args, span);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(PrimCallNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimCallNode>([](const ObjectRef& node, ReprPrinter* p) {
      MXCHECK(false) << "unimpl";
      //      auto* op = static_cast<const PrimCallNode*>(node.get());
      //      if (auto* ptr_op = op->op.as<OpNode>()) {
      //        p->stream << ptr_op->name << "(";
      //      } else {
      //        auto* ptr_gvar = op->op.as<GlobalVarNode>();
      //        CHECK(ptr_gvar != nullptr);
      //        p->stream << "@" << ptr_gvar->name_hint << "(";
      //      }
      //      for (size_t i = 0; i < op->args.size(); ++i) {
      //        p->Print(op->args[i]);
      //        if (i < op->args.size() - 1) {
      //          p->stream << ", ";
      //        }
      //      }
      //      p->stream << ")";
    });

MATXSCRIPT_REGISTER_GLOBAL("runtime.GetIntImm").set_body_typed([](IntImm i) { return i->value; });

MATXSCRIPT_REGISTER_GLOBAL("runtime.GetFloatImm").set_body_typed([](FloatImm f) {
  return f->value;
});

}  // namespace ir
}  // namespace matxscript
