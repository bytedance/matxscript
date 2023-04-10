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

  template <typename T>
  void GenLinalgArithStatement(const std::string& arith_type,
                               const PrimBinaryOpNode<T>* op,
                               std::ostream& os);

  std::pair<std::string, std::string> GetNodeDataType(const PrimExprNode* op);

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

  std::unordered_map<const Object*, uint32_t> expr_index_map_;
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

std::pair<std::string, std::string> LinalgTextPrinter::GetNodeDataType(const PrimExprNode* op) {
  auto val_type_iter = val_type_map_.find(op);
  if (val_type_iter != val_type_map_.end()) {
    return val_type_iter->second;
  }
  std::string arith_suffix = "";
  std::string data_type = "";
  MXCHECK(op->dtype.lanes() == 1) << " lanes must be 1, but receive " << op->dtype.lanes();
  auto op_dtype = op->dtype.code();
  auto bits = op->dtype.bits();
  switch (op->dtype.code()) {
    case kDLInt: {
      arith_suffix = "i";
      if (bits == 8 || bits == 16 || bits == 32 || bits == 64) {
        data_type = "i" + std::to_string(bits);
      }
      case kDLFloat: {
        arith_suffix = "f";
        if (bits == 16 || bits == 32 || bits == 64) {
          data_type = "f" + std::to_string(bits);
        }
      }
    }
  }
  if (arith_suffix == "" || data_type == "") {
    MXCHECK(false) << "data type not supported, type: " << op->dtype.code()
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
  if (op->a->template IsInstance<PrimVarNode>()) {
    os << op->a << ", %";
  } else {
    os << expr_index_map_[op->a.get()] << ", %";
  }
  if (op->b->template IsInstance<PrimVarNode>()) {
    os << op->b << " : " << data_type << "\n";
  } else {
    os << expr_index_map_[op->b.get()] << " : " << data_type << "\n";
  }
  expr_index_map_[op] = cur_index_;
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
  if (op->value->IsInstance<HLOExprNode>()) {
    HLOExprFunctor::VisitExpr(runtime::Downcast<HLOExpr>(op->value), os);
    os << "return %" << expr_index_map_[op->value.get()];
  } else if (op->value->IsInstance<PrimExprNode>()) {
    PrimExprFunctor::VisitExpr(runtime::Downcast<PrimExpr>(op->value), os);
    os << "return %" << expr_index_map_[op->value.get()];
  } else {
    MXCHECK(false) << "[linalg] not support expr node: " << op->value;
  }
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
  // TODO, add func name and params
  VisitStmt(op->body, os);
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
