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
 * \file linalg_printer.h
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "matxscript/ir/base.h"
#include "matxscript/ir/prim_expr.h"
#include "matxscript/ir/prim_ops.h"
#include "matxscript/ir/prim_var.h"
#include "matxscript/ir/printer/kernel_printer/linalg_generic_printer.h"
#include "matxscript/ir/tensor_stmt.h"
#include "matxscript/ir/type.h"
#include "matxscript/runtime/dlpack.h"
#include "matxscript/runtime/object.h"

#include <matxscript/ir/expr_functor.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/stmt_functor.h>
#include <matxscript/ir/type_functor.h>
#include <matxscript/runtime/data_type.h>

namespace matxscript {
namespace ir {
namespace printer {

using namespace ::matxscript::ir;
using namespace ::matxscript::runtime;

void MLIRTextPrinter::NewScope() {
  var_name_scope.emplace_back(expr_name_map_->begin(), expr_name_map_->end());
  expr_name_map_ = &(var_name_scope.back());
  var_type_scope.emplace_back(val_type_map_->begin(), val_type_map_->end());
  val_type_map_ = &(var_type_scope.back());
}

void MLIRTextPrinter::PopScope() {
  var_name_scope.pop_back();
  expr_name_map_ = &(var_name_scope.back());
  var_type_scope.pop_back();
  val_type_map_ = &(var_type_scope.back());
}

// Error Handlers
void MLIRTextPrinter::VisitExprDefault_(const Object* op, std::ostream& os) {
  MXTHROW << "[MLIRTextPrinter] Unsupported Expr: " << op->GetTypeKey();
}

void MLIRTextPrinter::VisitStmtDefault_(const Object* op, std::ostream& os) {
  MXTHROW << "[MLIRTextPrinter] Unsupported Stmt: " << op->GetTypeKey();
}

void MLIRTextPrinter::VisitTypeDefault_(const Object* op, std::ostream& os) {
  MXTHROW << "[MLIRTextPrinter] Unsupported Type: " << op->GetTypeKey();
}

// Begin Expr
void MLIRTextPrinter::VisitExpr_(const IntImmNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const FloatImmNode* op, std::ostream& os) {
}

std::string MLIRTextPrinter::ConvertTypeToMLIR(const runtime::DataType& type) const {
  std::string data_type;
  auto const bits = type.bits();
  auto const type_code = type.code();

  switch (type_code) {
    case DataType::TypeCode::kInt: {
      data_type = "i" + std::to_string(bits);
      break;
    }
    case DataType::TypeCode::kUInt: {
      data_type = "ui" + std::to_string(bits);
      break;
    }
    case DataType::TypeCode::kFloat: {
      data_type = "f" + std::to_string(bits);
      break;
    }
    default: {
      MXTHROW << "data type not supported, type: " << type_code << " bits: " << bits;
    }
  }
  return data_type;
}

std::string MLIRTextPrinter::ConvertTypeToMLIR(const matxscript::ir::Type& type) const {
  if (auto* n = type.as<PrimTypeNode>()) {
    return ConvertTypeToMLIR(n->dtype);
  } else if (auto* n = type.as<PointerTypeNode>()) {
    auto dtype = ConvertTypeToMLIR(n->element_type);
    return "memref<?x" + dtype + ">";
  } else {
    MXTHROW << "Type " << type << " does not have a corresponding runtime::DataType";
    return "";
  }
}

std::string MLIRTextPrinter::ConvertTypeToMLIR(const matxscript::ir::Buffer& buffer) const {
  std::stringstream ss;
  ss << "memref<";
  for (auto dim : buffer->shape) {
    if (dim->IsInstance<PrimVarNode>()) {
      auto node = runtime::Downcast<PrimVar>(dim);
      if (expr_name_map_->find(dim.get()) == expr_name_map_->end()) {
        MXTHROW << "Buffer(" << buffer->name << ") is annotated with " << node->name_hint
                << ", but for now linalg printer only supports constant or predefined symbols";
      }
      ss << "?x";
    } else if (dim->IsInstance<IntImmNode>()) {
      auto node = runtime::Downcast<IntImm>(dim);
      ss << node->value << 'x';
    } else {
      MXTHROW << "Buffer(" << buffer->name << ") is annotated with " << dim->checked_type()
              << ", but for now linalg printer only supports constant or predefined symbols";
    }
  }
  ss << ConvertTypeToMLIR(buffer->dtype);
  ss << ">";
  return ss.str();
}

void MLIRTextPrinter::PrintNodeName(const BaseExpr& ptr, std::ostream& os) {
  if (expr_name_map_->find(ptr.get()) != expr_name_map_->end()) {
    os << expr_name_map_->at(ptr.get());
    return;
  }
  if (ptr->IsInstance<PrimVarNode>()) {
    auto node = runtime::Downcast<PrimVar>(ptr);
    expr_name_map_->emplace(ptr.get(), node->name_hint.c_str());
    os << node->name_hint;
    return;
  }
  MXTHROW << "Expr: " << ptr << " has no corrresponding ssa value";
}

std::pair<std::string, std::string> MLIRTextPrinter::GetNodeDataType(const PrimExprNode* op) {
  auto val_type_iter = val_type_map_->find(op);
  if (val_type_iter != val_type_map_->end()) {
    return val_type_iter->second;
  }
  std::string arith_suffix = "";
  std::string data_type = ConvertTypeToMLIR(op->checked_type());
  MXCHECK(op->dtype.lanes() == 1) << " lanes must be 1, but receive " << op->dtype.lanes();
  auto op_dtype = op->dtype.code();
  auto bits = op->dtype.bits();
  switch (op->dtype.code()) {
    case kDLInt:
      arith_suffix = "i";
      break;
    case kDLFloat:
      arith_suffix = "f";
      break;
    default:
      MXTHROW << "data type not supported, type: " << op->dtype.code() << " bits: " << bits;
  }

  if (arith_suffix == "" || data_type == "") {
    MXTHROW << "data type not supported, type: " << op->dtype.code()
            << " bits: " << op->dtype.bits();
  }

  auto node_data_type =
      std::make_pair<std::string, std::string>(std::move(data_type), std::move(arith_suffix));
  val_type_map_->emplace(op, node_data_type);
  return node_data_type;
}

template <typename T>
void MLIRTextPrinter::GenMLIRArithStatement(const std::string& arith_type,
                                            const PrimBinaryOpNode<T>* op,
                                            std::ostream& os) {
  PrimExprFunctor::VisitExpr(op->a, os);
  PrimExprFunctor::VisitExpr(op->b, os);
  auto data_arith_suffix_type = GetNodeDataType(op);
  std::string data_type = data_arith_suffix_type.first;
  std::string arith_total_type = arith_type + data_arith_suffix_type.second;

  os << "%" << cur_index_ << " = arith." << arith_total_type << " %";
  PrintNodeName(op->a, os);
  os << ", %";
  PrintNodeName(op->b, os);
  os << " : " << data_type << "\n";
  if (expr_name_map_->find(op) != expr_name_map_->end()) {
    MXTHROW << "[linalg] op is already in expr_index_map_";
  }
  expr_name_map_->emplace(op, std::to_string(cur_index_));
  cur_index_ += 1;
}

void MLIRTextPrinter::VisitExpr_(const PrimAddNode* op, std::ostream& os) {
  GenMLIRArithStatement("add", op, os);
}

void MLIRTextPrinter::VisitExpr_(const PrimSubNode* op, std::ostream& os) {
  GenMLIRArithStatement("sub", op, os);
}

void MLIRTextPrinter::VisitExpr_(const PrimMulNode* op, std::ostream& os) {
  GenMLIRArithStatement("mul", op, os);
}

void MLIRTextPrinter::VisitExpr_(const PrimDivNode* op, std::ostream& os) {
  GenMLIRArithStatement("div", op, os);
}

void MLIRTextPrinter::VisitExpr_(const PrimModNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimFloorDivNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimFloorModNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimMinNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimMaxNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimEQNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimNENode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimLTNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimLENode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimGTNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimGENode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimAndNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimOrNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimNotNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimSelectNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimVarNode* op, std::ostream& os) {
  // print nothing here
}

void MLIRTextPrinter::VisitExpr_(const PrimLetNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimCallNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const PrimCastNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  expr_name_map_->emplace(op, expr_name_map_->at(op->buffer->data.get()));
}

// Begin Stmt

void MLIRTextPrinter::VisitStmt_(const BufferStoreNode* op, std::ostream& os) {
  PrimExprFunctor::VisitExpr(op->value, os);
  std::string result = '%' + std::to_string(cur_index_);
  cur_index_++;
  os << result << " = " << '%' << expr_name_map_->at(op->value.get()) << ": "
     << MLIRTextPrinter::ConvertTypeToMLIR(op->buffer->dtype) << std::endl;
  os << "linalg.yield " << result << " : " << MLIRTextPrinter::ConvertTypeToMLIR(op->buffer->dtype)
     << std::endl;
}

void MLIRTextPrinter::VisitStmt_(const AllocaVarStmtNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitStmt_(const AssignStmtNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitStmt_(const ReturnStmtNode* op, std::ostream& os) {
  if (op->value->IsInstance<PrimExprNode>()) {
    auto node = runtime::Downcast<PrimExpr>(op->value);
    PrimExprFunctor::VisitExpr(node, os);
    os << "func.return %";
    PrintNodeName(op->value, os);
    os << " :";
    VisitType(node->checked_type(), os);
    os << std::endl;
  } else {
    MXTHROW << "[linalg] not support expr node: " << op->value;
  }
}

void MLIRTextPrinter::VisitStmt_(const AssertStmtNode* op, std::ostream& os) {
  // linalg does not support assert
  VisitStmtDefault_(op, os);
}

void MLIRTextPrinter::VisitStmt_(const IfThenElseNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitStmt_(const SeqStmtNode* op, std::ostream& os) {
  for (auto& stmt : op->seq) {
    VisitStmt(stmt, os);
  }
}

void MLIRTextPrinter::VisitStmt_(const ForNode* op, std::ostream& os) {
  // for is a parallel scope.
  VisitStmtDefault_(op, os);
}

void MLIRTextPrinter::VisitStmt_(const ExprStmtNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitStmt_(const PrimFuncNode* op, std::ostream& os) {
  // TODO, add func name and params
  os << "func.func @" << op->GetGlobalName();
  os << "(";
  auto func_params = op->GetParams();
  for (int i = 0; i < func_params.size(); i++) {
    auto& param = func_params[i];
    if (param->IsInstance<PrimVarNode>()) {
      auto node = runtime::Downcast<PrimVar>(param);
      os << "%" << node->name_hint << ": ";
      VisitType(node->checked_type(), os);
      expr_name_map_->emplace(param.get(), node->name_hint.c_str());
    } else {
      MXTHROW << "[linalg] not support arg node: " << param->checked_type();
    }
    if (i != func_params.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
  auto rt_type = op->GetReturnType();
  if (!IsVoidType(rt_type)) {
    os << "->";
    VisitType(rt_type, os);
  }
  // check if none
  // if so skip
  // otherwise ->retype
  os << "{" << std::endl;
  VisitStmt(op->body, os);
  os << "}" << std::endl;
}

void MLIRTextPrinter::VisitStmt_(const ComputeBlockNode* op, std::ostream& os) {
  LinalgGenericPrinter printer(this);
  printer.ComputeBlockToLinalgGeneric(op, os);
}
void MLIRTextPrinter::VisitStmt_(const ComputeBlockRealizeNode* op, std::ostream& os) {
}

// Begin Type
void MLIRTextPrinter::VisitType_(const PrimTypeNode* node, std::ostream& os) {
  os << ConvertTypeToMLIR(node->dtype);
}

void MLIRTextPrinter::VisitType_(const PointerTypeNode* node, std::ostream& os) {
  auto dtype = ConvertTypeToMLIR(node->element_type);
  os << "memref<?x" + dtype + ">";
}

void MLIRTextPrinter::VisitType_(const DynTensorTypeNode* node, std::ostream& os) {
  VisitTypeDefault_(node, os);
}

// Global Linalg TextPrint
void MLIRTextPrinter::AddFunction(const PrimFunc& fn) {
  this->VisitStmt_(fn.get(), stream_);
}

StringRef MLIRTextPrinter::Finish() {
  return {stream_.str()};
}

MATXSCRIPT_REGISTER_GLOBAL("node.as_linalg_text").set_body_typed([](const PrimFunc& fn) {
  MLIRTextPrinter printer;
  printer.AddFunction(fn);
  return printer.Finish();
});

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
