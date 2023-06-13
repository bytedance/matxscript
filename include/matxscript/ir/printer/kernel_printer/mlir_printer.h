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
 * \file mlir_printer.h
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */
#pragma once
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
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
class LinalgGenericPrinter;
class MLIRTextPrinter : public StmtFunctor<void(const Stmt&, std::ostream&)>,
                        public PrimExprFunctor<void(const PrimExpr&, std::ostream&)>,
                        public HLOExprFunctor<void(const HLOExpr&, std::ostream&)>,
                        public TypeFunctor<void(const Type&, std::ostream&)> {
  using expr_name_map = std::unordered_map<const Object*, std::string>;
  using var_name_map = std::unordered_map<StringRef, std::string>;
  using var_type_map = std::unordered_map<const Object*, std::pair<std::string, std::string>>;
  friend class LinalgGenericPrinter;

 public:
  explicit MLIRTextPrinter() : expr_name_scope(1), var_type_scope(1), var_name_scope(1) {
    expr_name_map_ = &(expr_name_scope.back());
    val_type_map_ = &(var_type_scope.back());
    var_name_map_ =  &(var_name_scope.back());
  }

  void AddFunction(const PrimFunc& fn);
  StringRef Finish();

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
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) override;
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
  void VisitStmt_(const AllocateNode* op, std::ostream& os) override;
  void VisitStmtDefault_(const Object* op, std::ostream& os) override;

  void VisitStmt_(const BufferStoreNode* op, std::ostream& os) override;
  void VisitStmt_(const ComputeBlockNode* op, std::ostream& os) override;
  void VisitStmt_(const ComputeBlockRealizeNode* op, std::ostream& os) override;

  template <typename T>
  void GenMLIRArithStatement(const std::string& arith_type,
                             const PrimBinaryOpNode<T>* op,
                             std::ostream& os,
                             const std::unordered_map<std::string, std::string>& suffix_map = {});

  template <typename T>
  void GenMLIRCompareStatement(const std::string& compare_type,
                               const PrimCmpOpNode<T>* op,
                               std::ostream& os);

  std::string ConvertTypeToMLIR(const runtime::DataType& type) const;
  std::string ConvertTypeToMLIR(const Type& type) const;
  std::string ConvertTypeToMLIR(const Buffer& buffer) const;
  std::string ConvertTypeToMLIR(const PointerTypeNode* node) const;
  void PrintNodeName(const BaseExpr& ptr, std::ostream& os);

  std::pair<std::string, std::string> GetNodeDataType(const PrimExprNode* op);

  // Begin Type
  // Overload of Type printing functions
  //------------------------------------
  void VisitType_(const PrimTypeNode* node, std::ostream& os) override;
  void VisitType_(const PointerTypeNode* node, std::ostream& os) override;
  void VisitType_(const DynTensorTypeNode* node, std::ostream& os) override;
  void VisitTypeDefault_(const Object* op, std::ostream& os) override;

  void NewScope();
  void PopScope();
  template <class... Args>
  void insert_or_assign_expr_name_map_(const Object* key, Args&&... args) {
    auto rt = expr_name_map_->emplace(
        std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple(args...));
    if (!rt.second) {
      rt.first->second = std::string(args...);
    }
  }

  template <class... Args>
  void insert_or_assign_var_name_map_(StringRef &key, Args&&... args) {
    auto rt = var_name_map_->emplace(
        std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple(args...));
    if (!rt.second) {
      rt.first->second = std::string(args...);
    }
  }

 private:
  /*! \brief the stream to be printed */
  std::ostringstream stream_;
  std::unordered_map<const PointerTypeNode*, const Buffer> pointer_buffer_map;
  std::unordered_map<const PrimExprNode*, const std::string> index_map;
  std::vector<expr_name_map> expr_name_scope;
  std::vector<var_type_map> var_type_scope;
  std::vector<var_name_map> var_name_scope;
  expr_name_map* expr_name_map_;
  var_type_map* val_type_map_;
  var_name_map* var_name_map_;
  std::atomic<uint32_t> cur_index_{0};
  std::unique_ptr<LinalgGenericPrinter> computeBlockPrinter = nullptr;
};

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
