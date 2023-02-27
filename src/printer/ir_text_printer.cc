// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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
 * \file tir_text_printer.cc
 * \brief Printer to print out the IR text format
 *        that can be parsed by a parser.
 */
#include "doc.h"
#include "text_printer.h"

#include <algorithm>
#include <string>

#include <matxscript/ir/expr.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/op_expr.h>
#include <matxscript/ir/stmt.h>
#include <matxscript/ir/type.h>
#include <matxscript/ir/type_functor.h>

namespace matxscript {
namespace printer {

Doc IRTextPrinter::Print(const ObjectRef& node) {
  if (!node.defined())
    return Doc::Text("(nullptr)");
  if (node->IsInstance<ArrayNode>()) {
    return PrintArray(node.as<ArrayNode>());
  } else if (node->IsInstance<StringNode>()) {
    return PrintString(node.as<StringNode>());
  } else if (node->IsInstance<ir::TupleNode>()) {
    return PrintTuple(node.as<ir::TupleNode>());
  } else if (node->IsInstance<PrimFuncNode>()) {
    return PrintFunc(Downcast<PrimFunc>(node));
  } else if (node->IsInstance<FunctionNode>()) {
    return PrintFunc(Downcast<Function>(node));
  } else if (node->IsInstance<StmtNode>()) {
    return VisitStmt(Downcast<Stmt>(node));
  } else if (node->IsInstance<PrimExprNode>()) {
    return PrimExprFunctor<Doc(const PrimExpr&)>::VisitExpr(Downcast<PrimExpr>(node));
  } else if (node->IsInstance<HLOExprNode>()) {
    return PrintExpr(Downcast<HLOExpr>(node));
  } else if (node->IsInstance<TypeNode>()) {
    return PrintType(Downcast<Type>(node));
  } else {
    // TODO(maxiandi) : NDArray/Map
    std::stringstream ss;
    ss << static_cast<const void*>(node.get());
    return Doc::StrLiteral(ss.str());
  }
}

//------------------------------------
// Overload of Type printing functions
//------------------------------------
Doc IRTextPrinter::PrintType(const Type& type) {
  auto it = memo_type_.find(type);
  if (it != memo_type_.end())
    return it->second;
  Doc printed_type;
  printed_type = VisitType(type);
  memo_type_[type] = printed_type;
  return printed_type;
}

//------------------------------------
// Overload of Expr printing functions
//------------------------------------
Doc IRTextPrinter::PrintExpr(const HLOExpr& expr) {
  // Exploit memoization to print GNF.
  // The first time we visit an expression, we need to allocate a temp var
  // for it. Every subsequent time we can just use its assigned variable.
  // This works since hashing uses pointer equality.

  auto it = memo_.find(expr);
  if (it != memo_.end()) {
    return it->second;
  }

  Doc printed_expr = HLOExprFunctor<Doc(const HLOExpr&)>::VisitExpr(expr);
  memo_[expr] = printed_expr;
  return printed_expr;
}

std::vector<Doc> IRTextPrinter::PrintFuncAttrs(const Attrs& attrs) {
  std::vector<Doc> docs;
  if (const auto* dict_attrs = attrs.as<DictAttrsNode>()) {
    for (const auto& k : dict_attrs->dict) {
      docs.push_back(Doc::StrLiteral(k.first) << ": " << Print(k.second));
    }
  }
  return docs;
}

Doc IRTextPrinter::PrintFunc(const Doc& prefix, const BaseFunc& base_func) {
  Doc doc;
  doc << prefix;
  doc << PrintFunc(base_func);
  return doc;
}

Doc IRTextPrinter::PrintFunc(const BaseFunc& base_func) {
  memo_var_.clear();
  const auto& signature = base_func->func_type_annotation();

  Doc doc;
  doc << base_func->GetReprName();
  // print type params
  if (const auto* fn_node = base_func.as<FunctionNode>()) {
    if (!fn_node->type_params.empty()) {
      doc << "[";
      std::vector<Doc> type_params;
      for (const TypeVar& tv : fn_node->type_params) {
        type_params.push_back(Doc::Text(tv->name_hint));
      }
      doc << Doc::Concat(type_params);
      doc << "]";
    }
  }
  // print captures
  if (const auto* fn_node = base_func.as<LambdaFunctionNode>()) {
    if (!fn_node->captures.empty()) {
      doc << "[";
      std::vector<Doc> type_params;
      for (auto& cap : fn_node->captures) {
        type_params.push_back(Print(cap));
      }
      doc << Doc::Concat(type_params);
      doc << "]";
    }
  }
  // print params and its type annotation
  doc << "(";
  std::vector<Doc> params;
  for (const auto& param : base_func->GetParams()) {
    params.push_back(AllocVar(param));
  }
  Doc sep;
  doc << PrintSep(params, Doc::Indent(9, Doc::Text(", ")));
  doc << ")";
  // print return type
  doc << " -> " << Print(signature->ret_type);
  // print attr
  auto func_attrs = PrintFuncAttrs(base_func->attrs);
  Doc attr_doc;
  attr_doc << Doc::NewLine() << "attr = {" << PrintSep(func_attrs, Doc::Text(", ")) << "}";
  doc << Doc::Indent(2, attr_doc);
  doc << PrintBody(base_func->GetBody());
  return doc;
}

Doc IRTextPrinter::PrintArray(const ArrayNode* op) {
  Doc doc;
  doc << '[';
  for (size_t i = 0; i < op->size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    doc << Print(op->at(i));
  }
  doc << ']';
  return doc;
}

Doc IRTextPrinter::PrintTuple(const ir::TupleNode* op) {
  Doc doc;
  doc << '(';
  for (size_t i = 0; i < op->fields.size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    doc << Print(op->fields[i]);
  }
  doc << ')';
  return doc;
}

Doc IRTextPrinter::VisitExprDefault_(const Object* op) {
  std::stringstream ss;
  ss << "Expr(addr: " << static_cast<const void*>(op) << ")";
  return Doc::StrLiteral(ss.str());
}

Doc IRTextPrinter::VisitStmtDefault_(const Object* op) {
  std::stringstream ss;
  ss << "Stmt(addr: " << static_cast<const void*>(op) << ")";
  return Doc::StrLiteral(ss.str());
}

Doc IRTextPrinter::VisitExpr_(const IntImmNode* op) {
  return PrintConstScalar<int64_t>(op->dtype, op->value);
}

Doc IRTextPrinter::VisitExpr_(const FloatImmNode* op) {
  return PrintConstScalar<double>(op->dtype, op->value);
}

Doc IRTextPrinter::VisitExpr_(const StringImmNode* op) {
  return Doc::StrLiteral(op->value);
}

Doc IRTextPrinter::VisitExpr_(const UnicodeImmNode* op) {
  return Doc::StrLiteral(op->value);
}

/*!
 * \brief special method to print out const scalar
 * \param dtype The data type
 * \param value The value to be printed.
 */
template <typename T>
Doc IRTextPrinter::ScalarLiteral(DataType dtype, const T& value) {
  std::ostringstream os;
  if (dtype == DataType::Int(32)) {
    os << value;
  } else if (dtype == DataType::Float(32)) {
    os << value << 'f';
  } else if (dtype == DataType::Float(64)) {
    os << value << "f64";
  } else if (dtype == DataType::Bool()) {
    return Doc::PyBoolLiteral(value != 0);
  } else {
    os << value;
  }
  return Doc::Text(os.str());
}

Doc IRTextPrinter::VisitExpr_(const PrimCastNode* op) {
  Doc doc;
  doc << "cast(" << PrintDType(op->dtype) << ", " << Print(op->value) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOCastPrimNode* op) {
  Doc doc;
  doc << "cast(" << PrintDType(op->dtype) << ", " << Print(op->value) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOVarNode* op) {
  HLOVar hlo_var = GetRef<HLOVar>(op);
  if (memo_var_.count(hlo_var)) {
    return memo_var_[hlo_var];
  } else {
    Doc val;
    val << op->name_hint() << "-undefined-ir";
    return val;
  }
}

Doc IRTextPrinter::VisitExpr_(const PrimVarNode* op) {
  PrimVar prim_var = GetRef<PrimVar>(op);
  if (memo_var_.count(prim_var)) {
    return memo_var_[prim_var];
  } else {
    Doc val;
    val << op->name_hint << "-undefined-ir";
    return val;
  }
}

Doc IRTextPrinter::VisitExpr_(const PrimLetNode* op) {
  Doc doc;
  doc << "let " << Print(op->var) << " = " << Print(op->value) << " in " << Print(op->body);
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const PrimCallNode* op) {
  Doc doc;
  if (auto* ptr_op = op->op.as<OpNode>()) {
    doc << "@" << Doc::Text(ptr_op->name) << "(";
  } else {
    MXCHECK(false) << "not OpNode";
    //    // TODO(bohan): Print out the name by he global var in the module.
    //    auto* op_gvar = op->op.as<GlobalVarNode>();
    //    CHECK(op_gvar != nullptr);
    //    doc << "@" << Doc::Text(op_gvar->name_hint) << "(";
  }
  std::vector<Doc> args;
  for (const auto& arg : op->args) {
    args.push_back(Print(arg));
  }
  doc << PrintSep(args, Doc::Text(", ")) << ", dtype=" << PrintDType(op->dtype) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const CallNode* op) {
  Doc doc;
  // visit args first so they are lifted before the op
  // this places op closer to its call site
  std::vector<Doc> args;
  for (const BaseExpr& arg : op->args) {
    args.push_back(Print(arg));
  }
  // TODO (matx4) : add PrintCallAttrs ?
  //  for (const Doc& d : PrintCallAttrs(op->attrs, op->op)) {
  //    args.push_back(d);
  //  }
  const auto* cons_node = op->op.as<ConstructorNode>();
  if (cons_node) {
    doc << cons_node->name_hint << "Constructor";
  } else {
    if (auto* ptr_op = op->op.as<OpNode>()) {
      doc << Doc::Text(ptr_op->name);
    } else {
      doc << Print(op->op);
    }
  }
  return doc << "(" << Doc::Concat(args) << ")";
}

Doc IRTextPrinter::VisitExpr_(const ConstructorNode* n) {
  Doc doc;
  doc << n->name_hint << "Constructor";
  if (in_adt_def_ && n->inputs.size() != 0) {
    doc << "(";
    std::vector<Doc> inputs;
    for (Type input : n->inputs) {
      inputs.push_back(Print(input));
    }
    doc << Doc::Concat(inputs) << ")";
  }
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const InitializerListNode* op) {
  Doc doc;
  doc << '(';
  for (size_t i = 0; i < op->fields.size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    doc << Print(op->fields[i]);
  }
  doc << ')';
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const InitializerDictNode* op) {
  Doc doc;
  doc << '{';
  auto itr = op->fields.begin();
  for (size_t i = 0; i < op->fields.size(); ++i, ++itr) {
    if (i != 0) {
      doc << ", ";
    }
    doc << Print((*itr).first);
    doc << ": ";
    doc << Print((*itr).second);
  }
  doc << '}';
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOIteratorNode* op) {
  Doc doc;
  doc << "HLOIterator(" << Print(op->container) << "." << Print(op->method) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const EnumAttrNode* op) {
  Doc doc;
  doc << "EnumAttr(" << Print(op->enum_str) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const ClassGetItemNode* op) {
  Doc doc;
  doc << "ClassGetItem(" << Print(op->self) << "." << op->attr->value << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const NoneExprNode* op) {
  Doc doc;
  doc << "NoneExpr";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOCastNode* op) {
  Doc doc;
  doc << "cast(" << PrintType(op->checked_type_) << ", " << Print(op->value) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOMoveNode* op) {
  Doc doc;
  doc << "move(" << Print(op->value) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOEnumerateNode* op) {
  Doc doc;
  doc << "enumerate(" << Print(op->value) << ", start=" << Print(op->start) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOZipNode* op) {
  Doc doc;
  doc << "zip(";
  for (auto i = 0; i < op->values.size(); ++i) {
    if (i > 0) {
      doc << ", ";
    }
    doc << Print(op->values[i]);
  }
  doc << ")";
  return doc;
}

#define MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(OpName, OpString) \
  Doc IRTextPrinter::VisitExpr_(const OpName* op) {                 \
    Doc doc;                                                        \
    doc << "(" << Print(op->a) << OpString;                         \
    doc << Print(op->b) << ")";                                     \
    return doc;                                                     \
  }

MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimAddNode, " + ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimSubNode, " - ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimMulNode, "*")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimDivNode, " / ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimModNode, " % ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimEQNode, " == ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimNENode, " != ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimLTNode, " < ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimLENode, " <= ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimGTNode, " > ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimGENode, " >= ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimAndNode, " && ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(PrimOrNode, " || ")

Doc IRTextPrinter::VisitExpr_(const PrimFloorDivNode* op) {
  Doc doc;
  doc << "floordiv(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const PrimFloorModNode* op) {
  Doc doc;
  doc << "floormod(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const PrimMinNode* op) {
  Doc doc;
  doc << "min(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const PrimMaxNode* op) {
  Doc doc;
  doc << "max(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const PrimNotNode* op) {
  Doc doc;
  doc << "!" << Print(op->a);
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const PrimSelectNode* op) {
  Doc doc;
  doc << "select(" << Print(op->condition) << ", " << Print(op->true_value) << ", "
      << Print(op->false_value);
  return doc;
}

MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOAddNode, " + ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOSubNode, " - ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOMulNode, "*")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOEqualNode, " == ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLONotEqualNode, " != ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOLessThanNode, " < ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOLessEqualNode, " <= ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOGreaterThanNode, " > ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOGreaterEqualNode, " >= ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOAndNode, " && ")
MATXSCRIPT_DECLARE_TIR_TEXT_PRINTER_BINOP(HLOOrNode, " || ")

Doc IRTextPrinter::VisitExpr_(const HLOFloorDivNode* op) {
  Doc doc;
  doc << "floordiv(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLOFloorModNode* op) {
  Doc doc;
  doc << "floormod(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc IRTextPrinter::VisitExpr_(const HLONotNode* op) {
  Doc doc;
  doc << "!" << Print(op->a);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const AllocaVarStmtNode* op) {
  Doc doc;
  doc << "alloca " << AllocVar(op->var);
  if (op->init_value.defined()) {
    doc << " = " << Print(op->init_value);
  }
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const AssignStmtNode* op) {
  Doc doc;
  doc << "assign " << Print(op->lhs) << " = " << Print(op->rhs);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const ReturnStmtNode* op) {
  Doc doc;
  doc << "return " << Print(op->value);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const LetStmtNode* op) {
  Doc doc;
  doc << "let " << Print(op->var) << " = " << Print(op->value) << Doc::NewLine() << Print(op->body);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const AttrStmtNode* op) {
  Doc doc;
  doc << "attr [" << Print(op->node) << "] " << Doc::StrLiteral(op->attr_key) << " = "
      << Print(op->value);
  if (op->body->IsInstance<SeqStmtNode>()) {
    doc << PrintBody(op->body);
  } else {
    doc << ";" << Doc::NewLine() << Print(op->body);
  }
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const AssertStmtNode* op) {
  Doc doc;
  doc << "assert(" << Print(op->condition) << ", " << Print(op->message) << ")" << Doc::NewLine()
      << Print(op->body);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const IfThenElseNode* op) {
  Doc doc;
  doc << "if " << Print(op->condition) << PrintBody(op->then_case);
  if (op->else_case.defined()) {
    doc << " else" << PrintBody(op->else_case);
  }
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const ExceptionHandlerNode* op) {
  Doc doc;
  doc << "catch (";
  if (op->e.defined()) {
    doc << Print(op->e);
  } else {
    doc << "...";
  }
  doc << ")" << PrintBody(op->body);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const TryExceptNode* op) {
  Doc doc;
  doc << "try " << PrintBody(op->body);
  for (auto& handler : op->handlers) {
    doc << VisitStmt(handler);
  }
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const RaiseNode* op) {
  Doc doc;
  doc << "raise";
  if (op->exc.defined()) {
    doc << " " << Print(op->exc);
  }
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const SeqStmtNode* op) {
  std::vector<Doc> stmts;
  Doc seq_doc, doc;
  for (Stmt stmt : op->seq) {
    seq_doc << Doc::NewLine() << Print(stmt);
  }
  doc << " {" << Doc::Indent(2, seq_doc) << Doc::NewLine() << "}";
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const EvaluateNode* op) {
  Doc doc;
  doc << Print(op->value);
  return doc;
}

inline const char* ForType2String(ForType t) {
  switch (t) {
    case ForType::Serial:
      return "serial";
    case ForType::Parallel:
      return "parallel";
    case ForType::Vectorized:
      return "vectorized";
    case ForType::Unrolled:
      return "unroll";
  }
  MXLOG(FATAL) << "Unknown ForType";
  return "Unknown";
}

Doc IRTextPrinter::VisitStmt_(const ForNode* op) {
  Doc doc;
  doc << "for (" << AllocVar(op->loop_var) << ", " << Print(op->min) << ", " << Print(op->max)
      << ", " << Print(op->step) << ")";
  if (op->for_type != ForType::Serial) {
    doc << " " << Doc::StrLiteral(ForType2String(op->for_type));
  }
  doc << PrintBody(op->body);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const AutoForNode* op) {
  Doc doc;
  doc << "AutoFor (";
  for (auto i = 0; i < op->loop_vars.size(); ++i) {
    if (i > 0) {
      doc << ", ";
    }
    doc << AllocVar(op->loop_vars[i]);
  }
  doc << " : " << Print(op->raw_container) << ")";
  doc << PrintBody(op->body);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const WhileNode* op) {
  Doc doc;
  doc << "while (" << Print(op->cond) << ")";
  doc << PrintBody(op->body);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const ContinueNode* op) {
  Doc doc;
  doc << "continue";
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const BreakNode* op) {
  Doc doc;
  doc << "break";
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const ExprStmtNode* op) {
  Doc doc;
  doc << Print(op->expr);
  return doc;
}

Doc IRTextPrinter::VisitStmt_(const HLOYieldNode* op) {
  Doc doc;
  doc << "yield ";
  doc << Print(op->symbol);
  return doc;
}

Doc IRTextPrinter::VisitType_(const PrimTypeNode* node) {
  Doc doc;
  doc << PrintDType(node->dtype);
  return doc;
}

Doc IRTextPrinter::VisitType_(const PointerTypeNode* node) {
  Doc doc;
  doc << "Pointer(" << Print(node->element_type) << ")";
  return doc;
}

Doc IRTextPrinter::VisitType_(const RangeTypeNode* node) {
  Doc doc;
  doc << "Range";
  return doc;
}

Doc IRTextPrinter::VisitType_(const TupleTypeNode* node) {
  std::vector<Doc> fields;
  for (Type field : node->fields) {
    fields.push_back(Print(field));
  }
  Doc doc;
  doc << "(" << Doc::Concat(fields);
  // conform to python tuple format (1,)
  if (node->fields.size() == 1) {
    doc << ",";
  }
  return doc << ")";
}

Doc IRTextPrinter::PrintDType(DataType dtype) {
  return Doc::Text(runtime::DLDataType2String(dtype));
}

Doc IRTextPrinter::VisitType_(const ObjectTypeNode* node) {
  Doc doc;
  doc << (node->is_view ? "AnyView" : "Any");
  return doc;
}

Doc IRTextPrinter::VisitType_(const UnicodeTypeNode* node) {
  Doc doc;
  doc << (node->is_view ? "unicode_view" : "Unicode");
  return doc;
}

Doc IRTextPrinter::VisitType_(const StringTypeNode* node) {
  Doc doc;
  doc << (node->is_view ? "string_view" : "String");
  return doc;
}

Doc IRTextPrinter::VisitType_(const ListTypeNode* node) {
  Doc doc;
  doc << "List";
  return doc;
}

Doc IRTextPrinter::VisitType_(const DictTypeNode* node) {
  Doc doc;
  doc << "Dict";
  return doc;
}

Doc IRTextPrinter::VisitType_(const SetTypeNode* node) {
  Doc doc;
  doc << "Set";
  return doc;
}

Doc IRTextPrinter::VisitType_(const IteratorTypeNode* node) {
  Doc doc;
  doc << Print(node->container_type) << "_Iterator";
  return doc;
}

Doc IRTextPrinter::VisitType_(const ExceptionTypeNode* node) {
  Doc doc;
  doc << Print(node->name);
  return doc;
}

Doc IRTextPrinter::VisitType_(const FileTypeNode* node) {
  Doc doc;
  doc << "File";
  return doc;
}

Doc IRTextPrinter::VisitType_(const NDArrayTypeNode* node) {
  Doc doc;
  doc << node->GetPythonTypeName().encode();
  return doc;
}

Doc IRTextPrinter::VisitType_(const ClassTypeNode* node) {
  Doc doc;
  doc << "ClassType(name: " << node->header->name_hint;
  if (node->base.defined()) {
    doc << ", base: " << node->base.as<ClassTypeNode>()->header->name_hint;
  }
  doc << ")";
  // TODO: fix var/func print
  return doc;
}

Doc IRTextPrinter::VisitType_(const UserDataTypeNode* node) {
  Doc doc;
  doc << "UserDataType";
  return doc;
}

Doc IRTextPrinter::VisitType_(const OpaqueObjectTypeNode* node) {
  Doc doc;
  doc << "OpaqueObjectType";
  return doc;
}

Doc IRTextPrinter::VisitType_(const RefTypeNode* node) {
  Doc doc;
  doc << "RefType(" << this->VisitType(node->value) << ")";
  return doc;
}

template <typename T>
Doc IRTextPrinter::PrintConstScalar(DataType dtype, const T& data) {
  Doc doc;
  std::ostringstream os;
  os << data;
  if (dtype == DataType::Int(32)) {
    doc << Doc::Text(os.str());
  } else {
    if (dtype.bits() == 1 && dtype.lanes() == 1 && dtype.code() == kDLUInt) {
      doc << ((data == 1) ? "True" : "False");
      return doc;
    }
    doc << Doc::Text(os.str());
    switch (dtype.code()) {
      case kDLInt:
        doc << "i";
        break;
      case kDLUInt:
        doc << "u";
        break;
      case kDLFloat:
        doc << "f";
        break;
    }
    doc << Doc::Text(std::to_string(dtype.bits()));
    if (dtype.lanes() != 1)
      doc << "x" << Doc::Text(std::to_string(dtype.lanes()));
  }
  return doc;
}

/*!
 * \brief get a unique name with the corresponding prefix
 * \param prefix The prefix of the name
 * \return The returned name.
 */
Doc IRTextPrinter::GetUniqueName(const runtime::String& prefix) {
  // std::replace(prefix.begin(), prefix.end(), '.', '_');
  runtime::String unique_prefix = prefix;
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end()) {
    while (true) {
      std::ostringstream os;
      os << prefix << (++it->second);
      runtime::String name = os.str();
      if (name_alloc_map_.count(name) == 0) {
        unique_prefix = name;
        break;
      }
    }
  }
  name_alloc_map_[unique_prefix] = 0;
  return Doc::Text(unique_prefix);
}

Doc IRTextPrinter::Print(TypeKind k) {
  switch (k) {
    case kType:
      return Doc::Text("Type");
    case kShapeVar:
      return Doc::Text("Shape");
    case kBaseType:
      return Doc::Text("BaseType");
    case kConstraint:
      return Doc::Text("Constraint");
    case kAdtHandle:
      return Doc::Text("AdtHandle");
    default:
      MXLOG(ERROR) << "Unknown Kind";
      throw;
  }
}

Doc IRTextPrinter::AllocVar(const BaseExpr& var_base) {
  // still print if ir is malformed, but show the error.
  if (memo_var_.count(var_base)) {
    Doc val = memo_var_[var_base];
    val << "-malformed-ir";
    return val;
  }
  if (var_base.as<PrimVarNode>()) {
    auto var = Downcast<PrimVar>(var_base);
    runtime::String name = var->name_hint;
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "v" + name;
    }
    Doc val = GetUniqueName(name);
    memo_var_[var] = val;
    return val << ": " << Print(GetType(var));
  } else {
    MXCHECK(var_base.as<HLOVarNode>()) << "var is not PrimVar or a HLOVar";
    auto var = Downcast<HLOVar>(var_base);
    runtime::String name = var->name_hint();
    // always make sure first name is alpha
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "v" + name;
    }
    Doc val = GetUniqueName("%" + name);
    memo_var_[var] = val;
    if (var->type_annotation.defined()) {
      val << ": " << Print(var->type_annotation);
    }
    return val;
  }
}

/*!
 * \brief Allocate name to a type variable.
 * \param var The input type variable.
 * \return The corresponding name.
 */
Doc IRTextPrinter::AllocTypeVar(const TypeVar& var) {
  if (memo_type_.count(var)) {
    Doc val = memo_type_[var];
    val << "-malformed-ir";
    return val;
  }
  runtime::String name = var->name_hint;
  if (name.length() == 0 || !std::isalpha(name[0])) {
    name = "t" + name;
  }
  Doc val = GetUniqueName(name);
  memo_type_[var] = val;
  if (var->kind != kType) {
    val << ": " << Print(var->kind);
  }
  return val;
}

Doc IRTextPrinter::PrintSep(const std::vector<Doc>& vec, const Doc& sep) {
  Doc seq;
  if (vec.size() != 0) {
    seq = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
      seq << sep << vec[i];
    }
  }
  return seq;
}

Doc IRTextPrinter::PrintBody(const ObjectRef& node, bool indent) {
  if (node->IsInstance<SeqStmtNode>()) {
    return Print(node);
  } else if (node->IsInstance<StmtNode>()) {
    Doc doc;
    doc << " {" << Doc::Indent(2, Doc::NewLine() << Print(node)) << Doc::NewLine() << "}";
    return doc;
  } else {
    Doc doc;
    Doc body;
    doc << "{";
    doc << Doc::Indent(indent, body << Doc::NewLine() << PrintScope(node)) << Doc::NewLine();
    doc << "}";
    return doc;
  }
}

// create a new scope by creating a new printer object. This allows temp var
// numbers to be reused and prevents hoisted vars from escaping too far
Doc IRTextPrinter::PrintScope(const ObjectRef& node) {
  // print in a new scope
  doc_stack_.push_back(Doc());
  // must print first so doc_stack_.back() reference doesn't become stale
  Doc doc = Print(node);
  doc = doc_stack_.back() << doc;
  doc_stack_.pop_back();
  return doc;
}

Doc IRTextPrinter::TempVar(int n) {
  Doc doc;
  return doc << "%" << n;
}

Doc IRTextPrinter::AllocTemp() {
  return TempVar(temp_var_counter_++);
}

}  // namespace printer
}  // namespace matxscript
