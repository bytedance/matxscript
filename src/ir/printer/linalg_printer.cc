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
#include <sstream>

#include <matxscript/ir/expr_functor.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/stmt_functor.h>
#include <matxscript/ir/type_functor.h>

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

void LinalgTextPrinter::VisitExpr_(const PrimAddNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimSubNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimMulNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitExpr_(const PrimDivNode* op, std::ostream& os) {
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
}

void LinalgTextPrinter::VisitStmt_(const AssertStmtNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const IfThenElseNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const SeqStmtNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const ForNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const ExprStmtNode* op, std::ostream& os) {
}

void LinalgTextPrinter::VisitStmt_(const PrimFuncNode* op, std::ostream& os) {
}

// Begin Type
void LinalgTextPrinter::VisitType_(const PrimTypeNode* node, std::ostream& os) {
}

void LinalgTextPrinter::VisitType_(const PointerTypeNode* node, std::ostream& os) {
}

void LinalgTextPrinter::VisitType_(const NDArrayTypeNode* node, std::ostream& os) {
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
