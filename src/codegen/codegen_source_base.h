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
 * \file codegen_source_base.h
 * \brief Common utilities to source code in text form.
 */
#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <matxscript/ir/expr.h>
#include <matxscript/ir/prim_ops.h>
#include <matxscript/runtime/module.h>

namespace matxscript {
namespace codegen {

/*!
 * \brief A base class to generate source code.
 * Contains helper utilities to generate nest and ssa form.
 */
class CodeGenSourceBase {
 public:
  virtual ~CodeGenSourceBase() = default;
  /*!
   * \brief Register constant value appeared in expresion tree
   *  This avoid generated a ssa id for each appearance of the value
   * \param value The constant value.
   */
  void MarkConst(runtime::String value);

 protected:
  /*! \brief entry in ssa assign map */
  struct SSAEntry {
    /*! \brief The value id */
    runtime::String vid;
    /*! \brief The scope id, used to check if this entry is invalid. */
    int scope_id;
  };
  /*! \brief Clear the states that might relates to function generation */
  void ClearFuncState();
  /*! \brief print the current indented value */
  void PrintIndent(std::ostream& os);
  /*!
   * \brief Allocate a variable name for a newly defined var.
   * \param v The variable.
   * \return the variable name.
   */
  runtime::String AllocVarID(const ir::PrimVarNode* v);
  runtime::String AllocVarID(const ir::HLOVarNode* v);
  runtime::String AllocVarID(const ir::BaseExpr& expr) {
    if (auto node_prim = expr.as<ir::PrimVarNode>()) {
      return AllocVarID(node_prim);
    } else if (auto node_hlo = expr.as<ir::HLOVarNode>()) {
      return AllocVarID(node_hlo);
    } else {
      MXCHECK(false) << "expr is not a var: " << expr;
    }
    return "this is a sb call";
  }
  /*!
   * \brief Get a variable name.
   * \param v The variable.
   * \return the variable name.
   */
  runtime::String GetVarID(const ir::PrimVarNode* v) const;
  runtime::String GetVarID(const ir::HLOVarNode* v) const;
  runtime::String GetVarID(const ir::BaseExpr& expr) const {
    if (auto node_prim = expr.as<ir::PrimVarNode>()) {
      return GetVarID(node_prim);
    } else if (auto node_hlo = expr.as<ir::HLOVarNode>()) {
      return GetVarID(node_hlo);
    } else {
      MXCHECK(false) << "expr is not a var: " << expr;
    }
    return "this is a sb call";
  }
  /*!
   * \brief Get the SSA ID corresponds to src
   *  If necessary, generate new assignment
   * \param src The source expression
   * \param t The type of the expression.
   */
  runtime::String SSAGetID(runtime::String src, ir::Type t, std::ostream& os);
  runtime::String SSAGetID(runtime::String src, runtime::DataType t, std::ostream& os) {
    return SSAGetID(std::move(src), ir::PrimType(t), os);
  }
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  runtime::String GetUniqueName(runtime::String prefix);
  /*!
   * \brief mark the beginning of a new scope
   * \return The scope id.
   */
  int BeginScope();
  /*!
   * \brief mark the end of an old scope.
   * \param scope_id The scope id to be ended.
   */
  void EndScope(int scope_id);
  /*!
   * \brief Print assignment of src to the id in ssa entry.
   * \param target id of target variable.
   * \param src The source expression.
   * \param t The type of target.
   */
  virtual void PrintSSAAssign(const runtime::String& target,
                              const runtime::String& src,
                              ir::Type t,
                              std::ostream& os) = 0;

  /*! \brief the declaration stream */
  std::ostringstream decl_stream;
  /*! \brief the stream to be printed */
  std::ostringstream stream;
  /*! \brief name of each variable */
  std::unordered_map<const ir::BaseExprNode*, runtime::String> var_idmap_;

 private:
  /*! \brief assignment map of ssa */
  std::unordered_map<runtime::String, SSAEntry> ssa_assign_map_;
  /*! \brief name allocation map */
  std::unordered_map<runtime::String, int> name_alloc_map_;
  /*! \brief array to check whether we are inside certain scope */
  std::vector<bool> scope_mark_;
  /*! \brief The current indentation value */
  int indent_{0};
};

/*!
 * \brief Create a source module for viewing.
 * \param code The code to be viewed.
 * \param fmt The code. format.
 */
runtime::Module SourceModuleCreate(runtime::String code, runtime::String fmt);

/*!
 * \brief Create a C source module for viewing and compiling GCC code.
 * \param code The code to be viewed.
 * \param fmt The code format.
 * \param symbol The symbol that the c source module represents.
 * \param const_vars. The constant variables that the c source module needs.
 * \return The created module.
 */
runtime::Module CSourceModuleCreate(const runtime::String& code,
                                    const runtime::String& fmt,
                                    const runtime::String& symbol = "",
                                    const runtime::Array<runtime::StringRef>& const_vars = {});

}  // namespace codegen
}  // namespace matxscript
