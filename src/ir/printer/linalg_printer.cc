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
#include "matxscript/ir/base.h"
#include "matxscript/ir/prim_expr.h"
#include "matxscript/ir/prim_ops.h"
#include "matxscript/ir/prim_var.h"
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

class LinalgTextPrinter : public StmtFunctor<void(const Stmt&, std::ostream&)>,
                          public PrimExprFunctor<void(const PrimExpr&, std::ostream&)>,
                          public HLOExprFunctor<void(const HLOExpr&, std::ostream&)>,
                          public TypeFunctor<void(const Type&, std::ostream&)> {
 public:
  explicit LinalgTextPrinter() {
  }

  void AddFunction(const PrimFunc& fn);
  StringRef Finish();

 private:
  // Begin Expr
  void VisitExpr_(const IntImmNode* op, std::ostream& os) override;
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimAddNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimSubNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimMulNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimDivNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimModNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimFloorDivNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimFloorModNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimMinNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimMaxNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimEQNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimNENode* op, std::ostream& os) override;
  void VisitExpr_(const PrimLTNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimLENode* op, std::ostream& os) override;
  void VisitExpr_(const PrimGTNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimGENode* op, std::ostream& os) override;
  void VisitExpr_(const PrimAndNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimOrNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimNotNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimSelectNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimVarNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimLetNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimCallNode* op, std::ostream& os) override;
  void VisitExpr_(const PrimCastNode* op, std::ostream& os) override;
  void VisitExprDefault_(const Object* op, std::ostream& os) override;

  // Begin Stmt
  void VisitStmt_(const AllocaVarStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const AssignStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const ReturnStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const AssertStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const IfThenElseNode* op, std::ostream& os) override;
  void VisitStmt_(const SeqStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const ForNode* op, std::ostream& os) override;
  void VisitStmt_(const ExprStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const PrimFuncNode* op, std::ostream& os) override;
  void VisitStmtDefault_(const Object* op, std::ostream& os) override;

  void VisitStmt_(const BufferStoreNode* op, std::ostream& os) override;
  void VisitStmt_(const ComputeBlockNode* op, std::ostream& os) override;
  void VisitStmt_(const ComputeBlockRealizeNode* op, std::ostream& os) override;

  template <typename T>
  void GenLinalgArithStatement(const std::string& arith_type,
                               const PrimBinaryOpNode<T>* op,
                               std::ostream& os);

  std::string ConvertTypeToMLIR(const runtime::DataType& type);
  std::string ConvertTypeToMLIR(const Type& type);
  std::string ConvertTypeToMLIR(const Buffer& buffer);
  void PrintNodeName(const BaseExpr& ptr, std::ostream& os);

  std::pair<std::string, std::string> GetNodeDataType(const PrimExprNode* op);

  void VisitRangeExpr_(const BufferRegion &buffer, const RangeExpr &rng, std::ostream &os);
  void GenAffineMap_(const Array<PrimIterVar>&iter_vars, const Array<BufferRegion> &reads, const Array<BufferRegion> &writes, std::ostream &os);
  void VisitBufferRegionArray_(const Array<BufferRegion> &reads, std::ostream& os);
  void VisitComputBlockBody_(const Stmt &body, std::ostream& os);

  void ComputeBlockToLinalgGeneric(const ComputeBlockNode* op, std::ostream& os);
  void LibraryNodeToLinalgGeneric();

  // Begin Type
  // Overload of Type printing functions
  //------------------------------------
  void VisitType_(const PrimTypeNode* node, std::ostream& os) override;
  void VisitType_(const PointerTypeNode* node, std::ostream& os) override;
  void VisitType_(const NDArrayTypeNode* node, std::ostream& os) override;
  void VisitTypeDefault_(const Object* op, std::ostream& os) override;

 private:
  /*! \brief the stream to be printed */
  std::ostringstream stream_;

  std::unordered_map<const Object*, std::string> expr_name_map_;
  std::unordered_map<const Object*, std::pair<std::string, std::string>> val_type_map_;
  std::atomic<uint32_t> cur_index_{0};
};

// Error Handlers
void LinalgTextPrinter::VisitExprDefault_(const Object* op, std::ostream& os) {
  MXTHROW << "[LinalgTextPrinter] Unsupported Expr: " << op->GetTypeKey();
}

void LinalgTextPrinter::VisitStmtDefault_(const Object* op, std::ostream& os) {
  MXTHROW << "[LinalgTextPrinter] Unsupported Stmt: " << op->GetTypeKey();
}

void LinalgTextPrinter::VisitTypeDefault_(const Object* op, std::ostream& os) {
  MXTHROW << "[LinalgTextPrinter] Unsupported Type: " << op->GetTypeKey();
}

// Begin Expr
void LinalgTextPrinter::VisitExpr_(const IntImmNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const FloatImmNode* op, std::ostream& os) {
}

std::string LinalgTextPrinter::ConvertTypeToMLIR(const runtime::DataType& type) {
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

std::string LinalgTextPrinter::ConvertTypeToMLIR(const matxscript::ir::Type& type) {
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

std::string LinalgTextPrinter::ConvertTypeToMLIR(const matxscript::ir::Buffer &buffer) {
  std::stringstream ss;
  ss << "memref<";
  for(auto dim : buffer->shape){
    if (dim->IsInstance<PrimVarNode>()){
      auto node = runtime::Downcast<PrimVar>(dim);
      if (expr_name_map_.find(dim.get())==expr_name_map_.end()){
        MXTHROW << "Buffer("<<buffer->name<<") is annotated with "<<node->name_hint
                <<", but for now linalg printer only supports constant or predefined symbols";
      }
      ss<<"?x";
    }else if (dim->IsInstance<IntImmNode>()){
      auto node = runtime::Downcast<IntImm>(dim);
      ss<<node->value<<'x';
    }else{
      MXTHROW << "Buffer("<<buffer->name<<") is annotated with "<<dim->checked_type()
              <<", but for now linalg printer only supports constant or predefined symbols";
    }
  }
  ss << ConvertTypeToMLIR(buffer->dtype);
  ss << ">";
  return ss.str();
}

void LinalgTextPrinter::PrintNodeName(const BaseExpr& ptr, std::ostream& os) {
  if (expr_name_map_.find(ptr.get()) != expr_name_map_.end()) {
    os << expr_name_map_.at(ptr.get());
    return;
  }
  if (ptr->IsInstance<PrimVarNode>()) {
    auto node = runtime::Downcast<PrimVar>(ptr);
    expr_name_map_.emplace(ptr.get(), node->name_hint.c_str());
    os << node->name_hint;
    return;
  }
  MXTHROW << "Expr: " << ptr << " has no corrresponding ssa value";
}

std::pair<std::string, std::string> LinalgTextPrinter::GetNodeDataType(const PrimExprNode* op) {
  auto val_type_iter = val_type_map_.find(op);
  if (val_type_iter != val_type_map_.end()) {
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
  val_type_map_[op] = node_data_type;
  return node_data_type;
}

template <typename T>
void LinalgTextPrinter::GenLinalgArithStatement(const std::string& arith_type,
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
  if (expr_name_map_.find(op) != expr_name_map_.end()) {
    MXTHROW << "[linalg] op is already in expr_index_map_";
  }
  expr_name_map_.emplace(op, std::to_string(cur_index_));
  cur_index_ += 1;
}

void LinalgTextPrinter::VisitExpr_(const PrimAddNode* op, std::ostream& os) {
  GenLinalgArithStatement("add", op, os);
}

void LinalgTextPrinter::VisitExpr_(const PrimSubNode* op, std::ostream& os) {
  GenLinalgArithStatement("sub", op, os);
}

void LinalgTextPrinter::VisitExpr_(const PrimMulNode* op, std::ostream& os) {
  GenLinalgArithStatement("mul", op, os);
}

void LinalgTextPrinter::VisitExpr_(const PrimDivNode* op, std::ostream& os) {
  GenLinalgArithStatement("div", op, os);
}

void LinalgTextPrinter::VisitExpr_(const PrimModNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimFloorDivNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimFloorModNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimMinNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimMaxNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimEQNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimNENode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimLTNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimLENode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimGTNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimGENode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimAndNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimOrNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimNotNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimSelectNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimVarNode* op, std::ostream& os) {
  // print nothing here
}

void LinalgTextPrinter::VisitExpr_(const PrimLetNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimCallNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimCastNode* op, std::ostream& os) {
}

// Begin Stmt
void LinalgTextPrinter::VisitStmt_(const AllocaVarStmtNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const AssignStmtNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const ReturnStmtNode* op, std::ostream& os) {
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

void LinalgTextPrinter::VisitStmt_(const AssertStmtNode* op, std::ostream& os) {
  // linalg does not support assert
  VisitStmtDefault_(op, os);
}

void LinalgTextPrinter::VisitStmt_(const IfThenElseNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const SeqStmtNode* op, std::ostream& os) {
  for (auto& stmt : op->seq) {
    VisitStmt(stmt, os);
  }
}

void LinalgTextPrinter::VisitStmt_(const ForNode* op, std::ostream& os) {
  // for is a parallel scope.
  VisitStmtDefault_(op, os);
}

void LinalgTextPrinter::VisitStmt_(const ExprStmtNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const PrimFuncNode* op, std::ostream& os) {
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
      expr_name_map_.emplace(param.get(), node->name_hint.c_str());
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

void LinalgTextPrinter::VisitStmt_(const BufferStoreNode* op, std::ostream& os) {
}
void LinalgTextPrinter::VisitStmt_(const ComputeBlockNode* op, std::ostream& os) {
  ComputeBlockToLinalgGeneric(op, os);
}
void LinalgTextPrinter::VisitStmt_(const ComputeBlockRealizeNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitBufferRegionArray_(const Array<matxscript::ir::BufferRegion> &arr_, std::ostream &os) {
  // region is ignored for now, and IMHO it should be ignored in this stage.
  std::stringstream types;
  for(int i=0; i<arr_.size(); i++){
    const auto & buffer = arr_[i]->buffer;
    const auto & region = arr_[i]->region;
    os<< buffer->data;
    types << ConvertTypeToMLIR(buffer);
    if (i!=arr_.size()-1){
      os << ", ";
      types << ", ";
    }
  }
  if(arr_.size()>0){
    os << ": " <<types.str();
  }
  return;
}


bool isInt(const matxscript::ir::PrimExpr &expr, const int expect){
  if (expr->IsInstance<IntImmNode>()){
    const auto &node = runtime::Downcast<IntImm>(expr);
    return node->value == expect;
  }
  return false;
}


void LinalgTextPrinter::VisitRangeExpr_(const matxscript::ir::BufferRegion &buffer, const matxscript::ir::RangeExpr &rng, std::ostream &os){
  const auto &start = rng->start;
  const auto &end = rng->stop;
  const auto &step = rng->step;
  // start has to be 0
  MXCHECK(isInt(step, 0))<<"The start ("<<start<< ") of range ("<<rng<<") of buffer ("<<buffer<<") is not 0";
  // step has to be 1
  MXCHECK(isInt(step, 1))<<"The step ("<<step<< ") of range ("<<rng<<") of buffer ("<<buffer<<") is not 1";
  // end
  if (end->IsInstance<PrimVarNode>()){
    const auto &node = runtime::Downcast<PrimVar>(end);
    // todo check if it is iter var
    os << node->name_hint;
  }else{
    MXTHROW<<"The end ("<<end<< ") of range ("<<rng<<") of buffer ("<<buffer<<") is not a iter var";
  }
}


void LinalgTextPrinter::GenAffineMap_(const Array<matxscript::ir::PrimIterVar> &iter_vars, const Array<matxscript::ir::BufferRegion> &reads, const Array<matxscript::ir::BufferRegion> &writes, std::ostream &os) {
  os <<"indexing_maps = [";

  // collect all iter vars and format them to affine_map<(i,j,k) -> (
  std::stringstream perfix;
  perfix << "affine_map<(";

  for (int i=0; i<iter_vars.size(); i++) {
    if(iter_vars[i]->dom->start->IsInstance<IntImmNode>()){
      const auto &node = runtime::Downcast<IntImm>(iter_vars[i]->dom->start);
      if(node->value!=0){
        MXTHROW<<"The start ("<<iter_vars[i]->dom->start<< ") of iter_var ("<<iter_vars[i]<<") is not 0";
      }
    }else{
      MXTHROW<<"The start ("<<iter_vars[i]->dom->start<< ") of iter_var ("<<iter_vars[i]<<")  is not 0";
    }
    if (!iter_vars[i]->dom->stop->IsInstance<PrimVarNode>()){
      MXTHROW<<"The end ("<<iter_vars[i]->dom->stop<< ") of iter_var ("<<iter_vars[i]<<") is not a pre defined symbol";
    }

    if (iter_vars[i]->dom->step->IsInstance<IntImmNode>()){
      const auto &node = runtime::Downcast<IntImm>(iter_vars[i]->dom->step);
      if(node->value!=1){
        MXTHROW<<"The step ("<<iter_vars[i]->dom->step<< ") of iter_var ("<<iter_vars[i]<<") is not 1";
      }
    }else{
      MXTHROW<<"The step ("<<iter_vars[i]->dom->step<< ") of iter_var ("<<iter_vars[i]<<")  is not 1";
    }
    perfix << iter_vars[i]->dom->stop;
    if (i!=reads.size()-1){
      perfix<<", ";
    }
  }
  perfix << ") -> (";
  auto perfix_str = perfix.str();

  // format for each ndarray
  for (const auto & read_buffer : reads) {
    os<<perfix_str;
    const auto &buffer = read_buffer->buffer;
    const auto &region = read_buffer->region;
    for(int i=0;i<region.size(); i++){
      const auto &range = region[i];
      VisitRangeExpr_(read_buffer, range, os);
      if (i!=region.size()-1){
        os<<", ";
      }
    }
    if (writes.empty()){
      os << ")>";
    }else {
      os << ")>, ";

    }
  }

  for (const auto & write_buffer : writes) {
    os<<perfix_str;
    const auto &buffer = write_buffer->buffer;
    const auto &region = write_buffer->region;
    for(int i=0;i<region.size(); i++){
      const auto &range = region[i];
      VisitRangeExpr_(write_buffer, range, os);
      if (i!=region.size()-1){
        os<<", ";
      }
    }
  }

  os << "], iterator_types = [";
  // todo for now just assume they are parallel, deal with reduction later
  for (int i=0; i<reads.size(); i++){
    os<<"parallel";
    if (i!=reads.size()-1){
      os<<", ";
    }
  }
  os << "]";
}

void LinalgTextPrinter::VisitComputBlockBody_(const matxscript::ir::Stmt &body, std::ostream &os) {

}

void LinalgTextPrinter::ComputeBlockToLinalgGeneric(const ComputeBlockNode* op, std::ostream& os) {
  /**
   *   Array<PrimIterVar> iter_vars;
   *   Array<BufferRegion> reads;
   *   Array<BufferRegion> writes;
   *   StringRef name_hint;
   *   Stmt body;
   */
  os<<"linalg.generic {";
  //visit iter_var (affine_map&iterator_types)
  GenAffineMap_(op->iter_vars, op->reads, op->writes, os);
  os<<"}"<<std::endl;
  //visit ins
  os<<"                    ins(";
  VisitBufferRegionArray_(op->reads, os);
  os<<')'<<std::endl;
  //visit outs
  os<<"                    outs(";
  VisitBufferRegionArray_(op->writes, os);
  os<<')'<<std::endl;
  os << "{"<<std::endl;
  // visit computblock
  VisitComputBlockBody_(op->body, os);
  os << "}"<<std::endl;
}
void LinalgTextPrinter::LibraryNodeToLinalgGeneric() {
}

// Begin Type
void LinalgTextPrinter::VisitType_(const PrimTypeNode* node, std::ostream& os) {
  os << ConvertTypeToMLIR(node->dtype);
}

void LinalgTextPrinter::VisitType_(const PointerTypeNode* node, std::ostream& os) {
  auto dtype = ConvertTypeToMLIR(node->element_type);
  os << "memref<?x" + dtype + ">";
}

void LinalgTextPrinter::VisitType_(const NDArrayTypeNode* node, std::ostream& os) {
  VisitTypeDefault_(node, os);
}

// Global Linalg TextPrint
void LinalgTextPrinter::AddFunction(const PrimFunc& fn) {
  this->VisitStmt_(fn.get(), stream_);
}

StringRef LinalgTextPrinter::Finish() {
  return {stream_.str()};
}

MATXSCRIPT_REGISTER_GLOBAL("node.as_linalg_text").set_body_typed([](const PrimFunc& fn) {
  LinalgTextPrinter printer;
  printer.AddFunction(fn);
  return printer.Finish();
});

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
