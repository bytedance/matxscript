// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the codegen is inspired by TVM.
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
 * \file codegen_c_host.h
 * \brief Generate C host code.
 */
#pragma once

#include "codegen_c.h"

#include <set>
#include <string>
#include <vector>

#include <matxscript/ir/expr.h>

namespace matxscript {
namespace codegen {

class CodeGenCHost final : public CodeGenC {
 public:
  CodeGenCHost();
  void Init(bool output_ssa, bool emit_asserts);
  void InitTypeRegistry(const ClassType& cls_ty);

  void BeginAnonymousNamespace();
  void EndAnonymousNamespace();

  void AddUserStructDeclaration(const ClassType& cls_ty);
  void AddUserStructInitDeclaration(const ClassType& cls_ty,
                                    const BaseFunc& init_func = BaseFunc(nullptr));
  void DefineUserStruct(const ClassType& cls_ty,
                        const std::unordered_map<String, BaseFunc>& methods);
  void DefineUserStructInitFunc(const ClassType& cls_ty,
                                const BaseFunc& init_func = BaseFunc(nullptr));

  void AddFunction(const BaseFunc& f);
  void AddFunctionDeclaration(const BaseFunc& f) override;

  void AddYieldFunction(const BaseFunc& f, const std::vector<HLOYield>& yield_stmts);

  void PrintType(DataType t, std::ostream& os) final;        // NOLINT(*)
  void PrintType(const Type& type, std::ostream& os) final;  // NOLINT(*)
  void PrintFuncPrefix(ir::Type ret_type) final;             // NOLINT(*)
  void PrintFinalReturn() final;                             // NOLINT(*)
  void PrintPackedFunctionMacro(const BaseFunc& f) final;
  void PrintPackedFunctionMacro(const String& global_symbol,
                                const String& bound_symbol,
                                const Type& ret_type,
                                const Array<BaseExpr>& args,
                                const Array<BaseExpr>& default_args,
                                bool first_arg_is_self,
                                bool capture_session_handle,
                                const Span& span) final;

  // overload visitor functions
  // void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const PrimCallNode* op, std::ostream& os) final;  // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void VisitExpr_(const PrimMinNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const PrimMaxNode* op, std::ostream& os) final;  // NOLINT(*)

  void VisitStmt_(const HLOYieldNode* op, std::ostream& os) final;  // NOLINT(*)

  void VisitExpr_(const ClassGetItemNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const NoneExprNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitStmt_(const LambdaFunctionNode* op, std::ostream& os) final;  // NOLINT(*)

  void VisitStmt_(const ExceptionHandlerNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitStmt_(const TryExceptNode* op, std::ostream& os) final;         // NOLINT(*)
  void VisitStmt_(const RaiseNode* op, std::ostream& os) final;             // NOLINT(*)

  /*! \brief Generate C runtime FuncRegistry global constant. */
  void GenerateFuncRegistry(const std::vector<String>& func_names, const String& class_name = "");

  /*! \brief Generate C runtime FuncRegistry global constant. */
  void GenerateClosuresNames(const std::vector<String>& func_names);

  /*! \brief Generate C runtime SystemLib entry point. */
  void GenerateCrtSystemLib();

 private:
  String module_name_;
  /* \brief tracks declared global variables which live despite GetUniqueName */
  std::set<String> declared_globals_;
  /* \brief names of the functions declared in this module */
  std::vector<String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;

  /*!
   * \brief Print ternary conditional operator implementing binary `op`
   * Forces the operands to be in SSA form.
   * \param op binary operator being expressed
   * \param compare string representation of comparison operator
   * \param os stream reference to print into
   */
  template <typename T>
  inline void PrintTernaryCondExpr(const T* op,
                                   const char* compare,
                                   std::ostream& os);  // NOLINT(*)
};

}  // namespace codegen
}  // namespace matxscript
