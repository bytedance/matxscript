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
 * \file codegen_c.h
 * \brief Common utilities to generated C style code.
 */
#pragma once

#include "codegen_source_base.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <matxscript/ir/analysis.h>
#include <matxscript/ir/expr.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/hlo_builtin.h>
#include <matxscript/ir/op_attr_types.h>
#include <matxscript/ir/prim_builtin.h>
#include <matxscript/ir/prim_ops.h>
#include <matxscript/ir/stmt.h>
#include <matxscript/ir/stmt_functor.h>
#include <matxscript/runtime/container.h>

namespace matxscript {
namespace codegen {

using namespace ir;
using namespace runtime;
/*!
 * \brief A base class to generate C code.
 *
 *  CodeGenC have two modes: generate SSA formed C code or normal form.
 *
 * **NOTE** CodeGenC does not aim at generating C codes consumed by MSVC or GCC,
 * Rather, it's providing infrastructural abstraction for C variants like CUDA
 * and OpenCL-C. You might find some odd variant features, e.g., type `int3` for
 * a vector of 3 `int`s. For native C code generator, see `CodeGenLLVM`.
 */
class CodeGenC : public PrimExprFunctor<void(const PrimExpr&, std::ostream&)>,
                 public HLOExprFunctor<void(const HLOExpr&, std::ostream&)>,
                 public StmtFunctor<void(const Stmt&, std::ostream&)>,
                 public CodeGenSourceBase {
  using HLOExprFunctor = HLOExprFunctor<void(const HLOExpr&, std::ostream&)>;
  using PrimExprFunctor = PrimExprFunctor<void(const PrimExpr&, std::ostream&)>;

 public:
  /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init(bool output_ssa);

  void PrintLineVars(std::ostream& os,
                     const Array<BaseExpr>& params,
                     const Array<BaseExpr>& default_params,
                     bool alloc_var,
                     bool with_var_name,
                     bool with_var_type = true,
                     bool with_defaults = false,
                     bool no_alias = false,
                     bool use_move = false,
                     bool skip_first = false);
  /*!
   * \brief Add the function to the generated module.
   * \param f The function to be compiled.
   * \param whether to append return 0 in the end.
   */
  virtual void AddFunction(const PrimFunc& f);
  virtual void AddFunctionDeclaration(const BaseFunc& f);

  virtual void AddFunction(const Function& f);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  String Finish();
  /*!
   * \brief Print the Stmt n to stream os
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n, std::ostream& os);

  virtual void VisitExpr(const BaseExpr& e, std::ostream& os) {
    if (e->IsInstance<HLOExprNode>()) {
      HLOExprFunctor::VisitExpr(runtime::Downcast<HLOExpr>(e), os);
    } else if (e->IsInstance<PrimExprNode>()) {
      PrimExprFunctor::VisitExpr(runtime::Downcast<PrimExpr>(e), os);
    } else {
      MXCHECK(false) << "[CodeGenC] not supported expr node: " << e;
    }
  }

  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const BaseExpr& n, std::ostream& os);
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  String PrintExpr(const BaseExpr& n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }
  // The following parts are overloadable print operations.
  /*!
   * \brief Print the function header before the argument list
   *
   *  Example: stream << "void";
   */
  virtual void PrintFuncPrefix(ir::Type ret_type);  // NOLINT(*)
  /*!
   * \brief Print the final return at the end the function.
   */
  virtual void PrintFinalReturn();  // NOLINT(*)
  /*!
   * \brief Insert statement before function body.
   * \param f The function to be compiled.
   */
  virtual void PreFunctionBody(const PrimFunc& f) {
  }
  virtual void PreFunctionBody(const Function& f) {
  }

  /*!
   * \brief Insert statement before function body.
   * \param f The function to be compiled.
   */
  virtual void PrintPackedFunctionMacro(const BaseFunc& f);
  virtual void PrintPackedFunctionMacro(const String& global_symbol,
                                        const String& bound_symbol,
                                        const Type& ret_type,
                                        const Array<BaseExpr>& args,
                                        const Array<BaseExpr>& default_args,
                                        bool first_arg_is_self,
                                        bool capture_session_handle,
                                        const Span& span);
  /*!
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState(const BaseFunc& f);
  virtual void InitAllState();

  // expression
  void VisitExpr_(const PrimVarNode* op, std::ostream& os) override;  // NOLINT(*)
  // void VisitExpr_(const PrimLoadNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimLetNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimCallNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const PrimAddNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimSubNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimMulNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimDivNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimFloorDivNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const PrimModNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimFloorModNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const PrimMinNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimMaxNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimEQNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const PrimNENode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const PrimLTNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const PrimLENode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const PrimGTNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const PrimGENode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const PrimAndNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimOrNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const PrimCastNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const HLOCastPrimNode* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const PrimNotNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const PrimSelectNode* op, std::ostream& os) override;    // NOLINT(*)

  // binary ops
  void VisitExpr_(const HLOAddNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const HLOSubNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const HLOMulNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const HLOFloorDivNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const HLOFloorModNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const HLOEqualNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const HLONotEqualNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const HLOLessThanNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const HLOLessEqualNode* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const HLOGreaterThanNode* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const HLOGreaterEqualNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const HLOAndNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const HLOOrNode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const HLONotNode* op, std::ostream& os) override;           // NOLINT(*)

  void VisitExpr_(const IntImmNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override;    // NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const UnicodeImmNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const InitializerListNode* op, std::ostream& os) override;
  // statement
  void VisitStmt_(const AllocaVarStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const AssignStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const ReturnStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const ForNode* op, std::ostream& os) override;
  void VisitStmt_(const AutoForNode* op, std::ostream& os) override;
  void VisitStmt_(const WhileNode* op, std::ostream& os) override;
  void VisitStmt_(const IfThenElseNode* op, std::ostream& os) override;
  void VisitStmt_(const AssertStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const EvaluateNode* op, std::ostream& os) override;
  void VisitStmt_(const SeqStmtNode* op, std::ostream& os) override;
  void VisitStmt_(const BreakNode* op, std::ostream& os) override;
  void VisitStmt_(const ContinueNode* op, std::ostream& os) override;
  void VisitStmt_(const ExprStmtNode* op, std::ostream& os) override;

  // hlo expression
  void PrintExplicitContainerBuiltinOp(const CallNode* op, std::ostream& os);  // NOLINT(*)
  void PrintGenericBuiltinOp(const CallNode* op, std::ostream& os);            // NOLINT(*)
  void PrintAsConstructor(const CallNode* op, std::ostream& os);
  void PrintConstructorValueType(const ConstructorNode* op, std::ostream& os);
  void PrintAsInitializeList(const InitializerListNode* op, std::ostream& os);
  void PrintAsInitializeDict(const InitializerDictNode* op, std::ostream& os);
  void PrintAsRTValue(BaseExpr expr, std::ostream& os);
  void VisitExpr_(const HLOVarNode* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const EnumAttrNode* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const HLOCastNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const HLOMoveNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const ConstructorNode* op, std::ostream& os) final;   // NOLINT(*)
  void VisitExpr_(const HLOEnumerateNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const HLOZipNode* op, std::ostream& os) final;        // NOLINT(*)

  /*!
   * Print source code span.
   * \param span The span representation.
   * \param os The stream to print the info into
   */
  virtual void PrintSpan(const Span& span, std::ostream& os);             // NOLINT(*);
  virtual void PrintSpanWithNewLine(const Span& span, std::ostream& os);  // NOLINT(*);
  virtual String GenPythonStyleSpanMessage(const Span& span,
                                           const string_view& func);  // NOLINT(*);

  /*!
   * Print Type represetnation of type t.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(runtime::DataType t, std::ostream& os);  // NOLINT(*)
  /*!
   * Print Type represetnation of type type.
   * \param type The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(const Type& type, std::ostream& os);  // NOLINT(*)

  virtual String PrintTypeAs(const String& value,
                             const String& type,
                             const String& py_info,
                             const String& value_repr);

  String PrintTypeAs(const String& value, const String& type, const String& py_info) {
    return PrintTypeAs(value, type, py_info, value);
  }
  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  // virtual void BindThreadIndex(const IterVar& iv);                             // NOLINT(*)
  virtual void PrintStorageScope(const String& scope, std::ostream& os);  // NOLINT(*)
  virtual void PrintStorageSync(const PrimCallNode* op);                  // NOLINT(*)
  // Binary vector op.
  virtual void PrintVecBinaryOp(const String& op,
                                DataType op_type,
                                PrimExpr lhs,
                                PrimExpr rhs,
                                std::ostream& os);  // NOLINT(*)
  // Get a cast type from to
  virtual String CastFromTo(String value, DataType from, DataType target);

 protected:
  // Print reference to a buffer as type t in index.
  virtual String GetBufferRef(DataType t, const PrimVarNode* buffer, PrimExpr index);

  /*!
   * \brief Handle volatile loads.
   *
   * This is to workaround a bug in CUDA cuda_fp16.h. Volatile accesses
   * to shared memory are required for reductions. However, __half class
   * does not implement volatile member functions. CUDA codegen will cast
   * away volatile qualifier from CUDA __half types.
   */
  //  virtual void HandleVolatileLoads(const String& value, const LoadNode* op, std::ostream&
  //  os) {
  //    // By default, do nothing but print the loaded value.
  //    os << value;
  //  }

  /*!
   * \brief Check if scope is part of type in the target language.
   *
   * **NOTE** In OpenCL, __local is part of type, so "__local int *"
   * is legal. This is not the case for CUDA, where "__shared__"
   * or "__constant__" is not part of type but a storage class (like
   * C/C++ static).
   */
  virtual bool IsScopePartOfType() const {
    return true;
  }

  /*!
   * \brief Print external function call.
   * \param ret_type The return type.
   * \param global_symbol The symbolc of the target function.
   * \param args The arguments to the function.
   * \param skip_first_arg Whether to skip the first arguments.
   * \param os The output stream.
   */
  virtual void PrintCallExtern(Type ret_type,
                               StringRef global_symbol,
                               const Array<BaseExpr>& args,
                               bool skip_first_arg,
                               std::ostream& os);  // NOLINT(*)

  virtual void PrintCallExtern(Type ret_type,
                               StringRef global_symbol,
                               const Array<PrimExpr>& args,
                               bool skip_first_arg,
                               std::ostream& os);  // NOLINT(*)
  /*!
   * \brief If buffer is allocated as type t.
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  bool HandleTypeMatch(const PrimVarNode* buf_var, DataType t) const;
  /*!
   * \brief Register the data type of buf_var
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  void RegisterHandleType(const PrimVarNode* buf_var, DataType t);
  // override
  void PrintSSAAssign(const String& target, const String& src, ir::Type t, std::ostream& os) final;
  /*! \brief reserves common C keywords */
  void ReserveKeywordsAsUnique();

  /*! \brief Check if buf_var is volatile or not. */
  bool IsVolatile(const PrimVarNode* buf_var) const {
    return volatile_buf_.count(buf_var) != 0;
  }

  /*! \brief restrict keyword */
  String restrict_keyword_{""};
  /*! \brief the storage scope of allocation */
  std::unordered_map<const PrimVarNode*, String> alloc_storage_scope_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const PrimVarNode*, DataType> handle_data_type_;
  /*! \brief Record of ops that have pre-defined global symbol. */
  OpAttrMap<TGlobalSymbol> op_attr_global_symbol_ = Op::GetAttrMap<TGlobalSymbol>("TGlobalSymbol");
  OpAttrMap<TGlobalIsGenericBuiltinOp> op_attr_is_generic_builtin_op_ =
      Op::GetAttrMap<TGlobalIsGenericBuiltinOp>("TGlobalIsGenericBuiltinOp");
  OpAttrMap<TGlobalIsExplicitContainerOp> op_attr_is_explicit_container_op_ =
      Op::GetAttrMap<TGlobalIsExplicitContainerOp>("TGlobalIsExplicitContainerOp");
  // cache commonly used ops
  const Op& builtin_call_extern_ = builtin::call_extern();
  const Op& builtin_call_pure_extern_ = builtin::call_pure_extern();

  String current_py_func_name_;
  ir::Type current_func_rt_type_;

 private:
  /*! \brief whether to print in SSA form */
  bool print_ssa_form_{false};
  /*! \brief set of volatile buf access */
  std::unordered_set<const PrimVarNode*> volatile_buf_;
  // deep comparison of PrimExpr
  ExprDeepEqual deep_equal_;
  // binding of let variables. Enables duplicate var defs that map to same value
  std::unordered_map<PrimVar, const PrimLetNode*, ObjectPtrHash, ObjectPtrEqual> let_binding_;
};

}  // namespace codegen
}  // namespace matxscript
