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
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "matxscript/ir/_base/cow_map_ref.h"
#include "matxscript/ir/base.h"
#include "matxscript/ir/none_expr.h"
#include "matxscript/ir/prim_expr.h"
#include "matxscript/ir/prim_ops.h"
#include "matxscript/ir/prim_var.h"
#include "matxscript/ir/printer/kernel_printer/linalg_generic_printer.h"
#include "matxscript/ir/tensor_stmt.h"
#include "matxscript/ir/type.h"
#include "matxscript/runtime/container/dict_private.h"
#include "matxscript/runtime/dlpack.h"
#include "matxscript/runtime/logging.h"
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
  os << '%' << cur_index_ << " = arith.constant " << std::to_string(op->value) << " : "
     << ConvertTypeToMLIR(op->checked_type()) << std::endl;
  if (expr_name_map_->find(op) != expr_name_map_->end()) {
    MXTHROW << "[linalg] op is already in expr_index_map_";
  }
  insert_or_assign_expr_name_map_(op, '%' + std::to_string(cur_index_));
  cur_index_ += 1;
}

void MLIRTextPrinter::VisitExpr_(const FloatImmNode* op, std::ostream& os) {
  os << '%' << cur_index_ << " = arith.constant " << std::to_string(op->value) << " : "
     << ConvertTypeToMLIR(op->checked_type()) << std::endl;
  if (expr_name_map_->find(op) != expr_name_map_->end()) {
    MXTHROW << "[linalg] op is already in expr_index_map_";
  }
  insert_or_assign_expr_name_map_(op, '%' + std::to_string(cur_index_));
  cur_index_ += 1;
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

std::string MLIRTextPrinter::ConvertTypeToMLIR(const matxscript::ir::PointerTypeNode* node) const {
  if (pointer_buffer_map.find(node) != pointer_buffer_map.end()) {
    return ConvertTypeToMLIR((pointer_buffer_map.at(node)));
  }
  MXTHROW << "Pointer type " << node->GetPythonTypeName() << " has not been binded to a buffer";
  return "";
}

std::string MLIRTextPrinter::ConvertTypeToMLIR(const matxscript::ir::Type& type) const {
  if (auto* n = type.as<PrimTypeNode>()) {
    return ConvertTypeToMLIR(n->dtype);
  } else if (auto* n = type.as<PointerTypeNode>()) {
    return ConvertTypeToMLIR(n);
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
        MXLOG(WARNING)
            << "[MLIRTextPrinter.ConvertTypeToMLIR] Buffer(" << buffer->name
            << ") is annotated with " << node->name_hint
            << ", which is not a constant or a predefined symbol. "
               "This could simply be that the buffer is used to initialize func parameters, "
               "which is a intended behavior. "
               "And for now it is treated as ? in MLIR.";
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
    insert_or_assign_expr_name_map_(ptr.get(), node->name_hint.c_str());
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
    case kDLUInt:
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

  os << '%' << cur_index_ << " = arith." << arith_total_type << " ";
  PrintNodeName(op->a, os);
  os << ", ";
  PrintNodeName(op->b, os);
  os << " : " << data_type << "\n";
  if (expr_name_map_->find(op) != expr_name_map_->end()) {
    MXTHROW << "[linalg] op is already in expr_index_map_";
  }
  insert_or_assign_expr_name_map_(op, '%' + std::to_string(cur_index_));
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

void printCastOp(const Type& origin, const Type& target, std::ostream& os) {
  /**
   * arith.extf - cast from floating-point to wider floating-point
   * arith.extsi - integer sign extension operation
   * arith.extui - integer zero extension operation
   * arith.fptosi - cast from floating-point type to signed integer type
   * arith.fptoui - cast from floating-point type to unsigned integer type
   * arith.sitofp - cast from signed integer type to floating-point
   * arith.truncf - cast from floating-point to narrower floating-point
   * arith.trunci - integer truncation operation
   * arith.uitofp - cast from unsigned integer type to floating-point
   */
  auto* origin_t = origin.as<PrimTypeNode>();
  auto* target_t = target.as<PrimTypeNode>();
  if (origin_t == nullptr || target_t == nullptr) {
    MXTHROW << "[MLIR] casting between non prim type node is not allowed";
  }
  auto const origin_t_bits = origin_t->dtype.bits();
  auto const origin_t_code = origin_t->dtype.code();
  auto const target_t_bits = target_t->dtype.bits();
  auto const target_t_code = target_t->dtype.code();
  if (origin_t_code != kDLInt && origin_t_code != kDLUInt && origin_t_code != kDLFloat) {
    MXTHROW << "[MLIR] the type being casted from is neither int nor float";
  }
  if (target_t_code != kDLInt && target_t_code != kDLUInt && target_t_code != kDLFloat) {
    MXTHROW << "[MLIR] the type being casted to is neither int nor float";
  }
  if (origin_t_code == target_t_code && origin_t_bits == target_t_bits) {
    MXTHROW << "[MLIR] casting between the same type is not allowed";
  }
  int compare = 0;
  if (target_t_bits < origin_t_bits) {
    compare = 0;
  } else if (target_t_bits == origin_t_bits) {
    compare = 1;
  } else {
    compare = 2;
  }
  /**
   *  {{{"i i t<o",  "i i t=o",  "i i t>o"},
   *    {"i ui t<o", "i ui t=o", "i ui t>o"},
   *    {"i f t<o",  "i f t=o",  "i f t>o"}},
   *
   *   {{"ui i t<o",  "ui i t=o",  "ui i t>o"},
   *    {"ui ui t<o", "ui ui t=o", "ui ui t>o"},
   *    {"ui f t<o",  "ui f t=o",  "ui f t>o"}},
   *
   *   {{"f i t<o",  "f i t=o",  "f i t>o"},
   *    {"f ui t<o", "f ui t=o", "f ui t>o"},
   *    {"f f t<o",  "f f t=o",  "f f t>o"}}}
   */

  static const std::string op_map[3][3][3] = {{{"arith.trunci", "arith.bitcast", "arith.extsi"},
                                               {"arith.trunci", "arith.bitcast", "arith.extui"},
                                               {"arith.sitofp", "arith.sitofp", "arith.sitofp"}},
                                              {{"arith.trunci", "arith.bitcast", "arith.extui"},
                                               {"arith.trunci", "arith.bitcast", "arith.extui"},
                                               {"arith.uitofp", "arith.uitofp", "arith.uitofp"}},
                                              {{"arith.fptosi", "arith.fptosi", "arith.fptosi"},
                                               {"arith.fptoui", "arith.fptoui", "arith.fptoui"},
                                               {"arith.truncf", "arith.bitcast", "arith.extf"}}};
  os << op_map[origin_t_code][target_t_code][compare] << ' ';
}

void MLIRTextPrinter::VisitExpr_(const PrimCastNode* op, std::ostream& os) {
  auto& v = op->value;
  PrimExprFunctor::VisitExpr(v, os);
  os << '%' << cur_index_ << " = ";
  printCastOp(v->checked_type(), op->checked_type(), os);
  PrintNodeName(v, os);
  os << " : ";
  os << ConvertTypeToMLIR(v->checked_type()) << " to " << ConvertTypeToMLIR(op->checked_type())
     << std::endl;
  if (expr_name_map_->find(op) != expr_name_map_->end()) {
    MXTHROW << "[linalg] op is already in expr_index_map_";
  }
  insert_or_assign_expr_name_map_(op, '%' + std::to_string(cur_index_));
  cur_index_ += 1;
}

void MLIRTextPrinter::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  insert_or_assign_expr_name_map_(op, computeBlockPrinter->GetPrimVarName(op));
}

// Begin Stmt

void MLIRTextPrinter::VisitStmt_(const BufferStoreNode* op, std::ostream& os) {
  PrimExprFunctor::VisitExpr(op->value, os);
  os << "linalg.yield " << expr_name_map_->at(op->value.get()) << " : "
     << MLIRTextPrinter::ConvertTypeToMLIR(op->buffer->dtype) << std::endl;
}

void MLIRTextPrinter::VisitStmt_(const AllocaVarStmtNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitStmt_(const AssignStmtNode* op, std::ostream& os) {
}

void MLIRTextPrinter::VisitStmt_(const ReturnStmtNode* op, std::ostream& os) {
  if (op->value->IsInstance<PrimExprNode>()) {
    auto node = runtime::Downcast<PrimExpr>(op->value);
    PrimExprFunctor::VisitExpr(node, os);
    os << "func.return ";
    PrintNodeName(op->value, os);
    os << " :";
    VisitType(node->checked_type(), os);
    os << std::endl;
  } else if (op->value->IsInstance<NoneExprNode>()) {
    os << "func.return " << std::endl;
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
  const auto& opt_parameter_map =
      op->GetAttr<matxscript::ir::Map<PrimVar, Buffer>>(attr::kKernelFunctionParameterBinding);
  if (!opt_parameter_map.defined()) {
    MXLOG(WARNING) << "Kernel function, " << op->GetReprName()
                   << ", is supposed to have an attribute named kKernelFunctionParameterBinding.";
  }
  const auto& parameter_map = opt_parameter_map.value();
  for (const auto& item : parameter_map) {
    const auto& var = item.first;
    const auto& buffer = item.second;
    const auto* var_type = var->checked_type().as<PointerTypeNode>();
    if (var_type == nullptr) {
      MXLOG(WARNING) << "The attribute, kKernelFunctionParameterBinding, binded to "
                     << op->GetReprName()
                     << ", is expected to have PrimVar of PointerType as keys.";
    }
    pointer_buffer_map.emplace(var_type, buffer);
  }

  os << "func.func @" << op->GetGlobalName();
  os << "(";
  auto func_params = op->GetParams();
  for (int i = 0; i < func_params.size(); i++) {
    auto& param = func_params[i];
    if (param->IsInstance<PrimVarNode>()) {
      auto node = runtime::Downcast<PrimVar>(param);
      os << '%' << node->name_hint << ": ";
      VisitType(node->checked_type(), os);
      insert_or_assign_expr_name_map_(param.get(), '%' + std::string(node->name_hint.data()));
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
  computeBlockPrinter = std::make_unique<LinalgGenericPrinter>(this);
  computeBlockPrinter->ComputeBlockToLinalgGeneric(op, os);
  computeBlockPrinter.reset();
}
void MLIRTextPrinter::VisitStmt_(const ComputeBlockRealizeNode* op, std::ostream& os) {
}

// Begin Type
void MLIRTextPrinter::VisitType_(const PrimTypeNode* node, std::ostream& os) {
  os << ConvertTypeToMLIR(node->dtype);
}

void MLIRTextPrinter::VisitType_(const PointerTypeNode* node, std::ostream& os) {
  os << ConvertTypeToMLIR(node);
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
