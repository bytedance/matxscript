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
 * \file text_printer.h
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */
#pragma once

#include "doc.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <matxscript/ir/expr_functor.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/module.h>
#include <matxscript/ir/stmt_functor.h>
#include <matxscript/ir/type_functor.h>

namespace matxscript {
namespace printer {

using namespace ::matxscript::ir;
using namespace ::matxscript::runtime;

class IRTextPrinter : public StmtFunctor<Doc(const Stmt&)>,
                      public PrimExprFunctor<Doc(const PrimExpr&)>,
                      public HLOExprFunctor<Doc(const HLOExpr&)>,
                      public TypeFunctor<Doc(const Type&)> {
 public:
  explicit IRTextPrinter() {
  }

  /*! \brief Print the node */
  Doc Print(const ObjectRef& node);

 private:
  /*! \brief Stack of docs to implement scoped GNFing. */
  std::vector<Doc> doc_stack_{};
  /*! \brief Map from Expr to Doc */
  std::unordered_map<HLOExpr, Doc, ObjectPtrHash, ObjectPtrEqual> memo_;
  /*! \brief Map from Var to Doc */
  std::unordered_map<BaseExpr, Doc, ObjectPtrHash, ObjectPtrEqual> memo_var_;
  /*! \brief Map from Type to Doc */
  std::unordered_map<Type, Doc, ObjectPtrHash, ObjectPtrEqual> memo_type_;
  /*! \brief name allocation map */
  std::unordered_map<runtime::String, int> name_alloc_map_;
  /*! \brief whether the printer is currently in an ADT definition */
  bool in_adt_def_;
  /*! \brief counter of temporary variable */
  size_t temp_var_counter_{0};

  // Begin prim constant
  Doc VisitExpr_(const IntImmNode* op) override;
  Doc VisitExpr_(const FloatImmNode* op) override;
  Doc VisitExpr_(const StringImmNode* op) override;
  Doc VisitExpr_(const UnicodeImmNode* op) override;
  // Begin arithmetic and logic ops
  Doc VisitExpr_(const PrimAddNode* op) override;
  Doc VisitExpr_(const PrimSubNode* op) override;
  Doc VisitExpr_(const PrimMulNode* op) override;
  Doc VisitExpr_(const PrimDivNode* op) override;
  Doc VisitExpr_(const PrimModNode* op) override;
  Doc VisitExpr_(const PrimFloorDivNode* op) override;
  Doc VisitExpr_(const PrimFloorModNode* op) override;
  Doc VisitExpr_(const PrimMinNode* op) override;
  Doc VisitExpr_(const PrimMaxNode* op) override;
  Doc VisitExpr_(const PrimEQNode* op) override;
  Doc VisitExpr_(const PrimNENode* op) override;
  Doc VisitExpr_(const PrimLTNode* op) override;
  Doc VisitExpr_(const PrimLENode* op) override;
  Doc VisitExpr_(const PrimGTNode* op) override;
  Doc VisitExpr_(const PrimGENode* op) override;
  Doc VisitExpr_(const PrimAndNode* op) override;
  Doc VisitExpr_(const PrimOrNode* op) override;
  Doc VisitExpr_(const PrimNotNode* op) override;
  Doc VisitExpr_(const PrimSelectNode* op) override;
  // Begin var/call...
  Doc VisitExpr_(const PrimVarNode* op) override;
  Doc VisitExpr_(const PrimLetNode* op) override;
  Doc VisitExpr_(const PrimCallNode* op) override;
  Doc VisitExpr_(const PrimCastNode* op) override;
  Doc VisitExpr_(const HLOCastPrimNode* op) override;
  // Begin HLO expr
  //------------------------------------
  // Overload of Expr printing functions
  //------------------------------------
  Doc PrintExpr(const HLOExpr& expr);
  template <typename T>
  static Doc ScalarLiteral(DataType dtype, const T& value);
  // Begin arithmetic and logic ops
  Doc VisitExpr_(const HLOAddNode* op) override;
  Doc VisitExpr_(const HLOSubNode* op) override;
  Doc VisitExpr_(const HLOMulNode* op) override;
  Doc VisitExpr_(const HLOFloorDivNode* op) override;
  Doc VisitExpr_(const HLOFloorModNode* op) override;
  Doc VisitExpr_(const HLOEqualNode* op) override;
  Doc VisitExpr_(const HLONotEqualNode* op) override;
  Doc VisitExpr_(const HLOLessThanNode* op) override;
  Doc VisitExpr_(const HLOLessEqualNode* op) override;
  Doc VisitExpr_(const HLOGreaterThanNode* op) override;
  Doc VisitExpr_(const HLOGreaterEqualNode* op) override;
  Doc VisitExpr_(const HLOAndNode* op) override;
  Doc VisitExpr_(const HLOOrNode* op) override;
  Doc VisitExpr_(const HLONotNode* op) override;

  Doc VisitExpr_(const CallNode* op) override;
  Doc VisitExpr_(const HLOVarNode* op) override;
  Doc VisitExpr_(const ConstructorNode* op) override;
  Doc VisitExpr_(const InitializerListNode* op) override;
  Doc VisitExpr_(const InitializerDictNode* op) override;
  Doc VisitExpr_(const HLOIteratorNode* op) override;
  Doc VisitExpr_(const EnumAttrNode* op) override;
  Doc VisitExpr_(const ClassGetItemNode* op) override;
  Doc VisitExpr_(const NoneExprNode* op) override;
  Doc VisitExpr_(const HLOCastNode* op) override;
  Doc VisitExpr_(const HLOMoveNode* op) override;
  Doc VisitExpr_(const HLOEnumerateNode* op) override;
  Doc VisitExpr_(const HLOZipNode* op) override;
  Doc VisitExpr_(const ir::TupleNode* op) override;
  Doc VisitExpr_(const ir::RangeExprNode* op) override;
  Doc VisitExprDefault_(const Object* op) override;

  // Begin stmt
  Doc VisitStmt_(const AllocaVarStmtNode* op) override;
  Doc VisitStmt_(const AssignStmtNode* op) override;
  Doc VisitStmt_(const ReturnStmtNode* op) override;
  Doc VisitStmt_(const LetStmtNode* op) override;
  Doc VisitStmt_(const AttrStmtNode* op) override;
  Doc VisitStmt_(const AssertStmtNode* op) override;
  Doc VisitStmt_(const IfThenElseNode* op) override;
  Doc VisitStmt_(const ExceptionHandlerNode* op) override;
  Doc VisitStmt_(const TryExceptNode* op) override;
  Doc VisitStmt_(const RaiseNode* op) override;
  Doc VisitStmt_(const SeqStmtNode* op) override;
  Doc VisitStmt_(const EvaluateNode* op) override;
  Doc VisitStmt_(const ForNode* op) override;
  Doc VisitStmt_(const AutoForNode* op) override;
  Doc VisitStmt_(const WhileNode* op) override;
  Doc VisitStmt_(const ContinueNode* op) override;
  Doc VisitStmt_(const BreakNode* op) override;
  Doc VisitStmt_(const ExprStmtNode* op) override;
  Doc VisitStmt_(const HLOYieldNode* op) override;
  Doc VisitStmtDefault_(const Object* op) override;

  // Begin Type
  // Overload of Type printing functions
  //------------------------------------
  Doc PrintType(const Type& type);
  Doc VisitType_(const PrimTypeNode* node) override;
  Doc VisitType_(const PointerTypeNode* node) override;
  Doc VisitType_(const RangeTypeNode* node) override;
  Doc VisitType_(const TupleTypeNode* node) override;
  Doc VisitType_(const ObjectTypeNode* node) override;
  Doc VisitType_(const UnicodeTypeNode* node) override;
  Doc VisitType_(const StringTypeNode* node) override;
  Doc VisitType_(const ListTypeNode* node) override;
  Doc VisitType_(const DictTypeNode* node) override;
  Doc VisitType_(const SetTypeNode* node) override;
  Doc VisitType_(const ExceptionTypeNode* node) override;
  Doc VisitType_(const IteratorTypeNode* node) override;
  Doc VisitType_(const FileTypeNode* node) override;
  Doc VisitType_(const NDArrayTypeNode* node) override;
  Doc VisitType_(const ClassTypeNode* node) override;
  Doc VisitType_(const UserDataTypeNode* node) override;
  Doc VisitType_(const OpaqueObjectTypeNode* node) override;
  Doc VisitType_(const RefTypeNode* node) override;

 public:
  std::vector<Doc> PrintFuncAttrs(const Attrs& attrs);
  Doc PrintFunc(const BaseFunc& base_func);
  Doc PrintFunc(const Doc& prefix, const BaseFunc& base_func);
  // Begin container
  Doc PrintArray(const ArrayNode* op);
  Doc PrintString(const StringNode* op) {
    return Doc::StrLiteral(op->data_container);
  }

  /*!
   * \brief special method to print out data type
   * \param dtype The data type
   */
  static Doc PrintDType(DataType dtype);
  /*!
   * \brief special method to print out const scalar
   * \param dtype The data type
   * \param data The pointer to hold the data.
   */
  template <typename T>
  static Doc PrintConstScalar(DataType dtype, const T& data);
  /*!
   * \brief Allocate name to a type variable.
   * \param var The input type variable.
   * \return The corresponding name.
   */
  Doc AllocTypeVar(const TypeVar& var);
  Doc AllocVar(const BaseExpr& var);
  /*!
   * \brief special method to render vectors of docs with a separator
   * \param vec vector of docs
   * \param sep separator
   */
  static Doc PrintSep(const std::vector<Doc>& vec, const Doc& sep);
  // indent a new body
  Doc PrintBody(const ObjectRef& body, bool indent = true);
  // create a new scope by creating a new printer object. This allows temp var
  // numbers to be reused and prevents hoisted vars from escaping too far
  Doc PrintScope(const ObjectRef& node);

  Doc TempVar(int n);
  Doc AllocTemp();
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  Doc GetUniqueName(const runtime::String& prefix);
  Doc Print(TypeKind k);
};

}  // namespace printer
}  // namespace matxscript

namespace matxscript {
namespace runtime {

class TextPrinter {
  using Doc = printer::Doc;

 public:
  explicit TextPrinter(const runtime::TypedNativeFunction<runtime::String(ObjectRef)>& annotate,
                       bool show_warning = true)
      : show_warning_(show_warning), annotate_(annotate), ir_text_printer_() {
  }

  /*! \brief whether show the meta data warning message */
  bool show_warning_;

  /*! \brief additional comment function */
  runtime::TypedNativeFunction<runtime::String(ObjectRef)> annotate_;
  /*! \brief IR Text Printer */
  printer::IRTextPrinter ir_text_printer_;

  Doc PrintFinal(const ObjectRef& node) {
    Doc doc;
    if (node->IsInstance<ir::IRModuleNode>()) {
      doc << PrintMod(Downcast<ir::IRModule>(node));
    } else {
      doc << ir_text_printer_.Print(node);
    }
    return doc;
  }

  Doc PrintMod(const ir::IRModule& mod);
};

}  // namespace runtime
}  // namespace matxscript
