/*
 * Acknowledgement: This file originates from TVM.
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
#pragma once

#include <matxscript/ir/analysis.h>
#include <matxscript/ir/prim_var.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/ir/stmt_functor.h>
#include <matxscript/ir/tensor_stmt.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace matxscript {
namespace ir {
namespace printer {

/*!
 * \brief Defines a variable in the IRDocsifier at the given frame,
 * and returns the corresponding IdDoc
 * \param var The variable to define
 * \param d The IRDocsifier
 * \param frame The frame to define the variable in
 * \return The IdDoc corresponding to the variable
 */
inline ExprDoc DefineVar(const ir::BaseExpr& var, const Frame& frame, const IRDocsifier& d) {
  if (Optional<ExprDoc> doc = d->GetVarDoc(var)) {
    return doc.value();
  }
  if (const auto* prim_var = var.as<ir::PrimVarNode>()) {
    return d->Define(var, frame, prim_var->name_hint.empty() ? "v" : prim_var->name_hint);
  }
  if (const auto* hlo_var = var.as<ir::HLOVarNode>()) {
    return d->Define(var, frame, hlo_var->name_hint().empty() ? "v" : hlo_var->name_hint());
  }
  MXTHROW << "[printer][DefineVar] the input expr is not a var!!!";
  return ExprDoc{nullptr};
}

/*!
 * \brief Recursively process the body statements of a TIR fragment represented by a frame
 * \param stmt The body statement to process
 * \param p The object path
 * \param f The frame
 * \param d The IRDocsifier
 */
inline void AsDocBody(const ir::Stmt& stmt, ObjectPath p, IRFrameNode* f, const IRDocsifier& d) {
  if (const auto* seq_stmt = stmt.as<ir::SeqStmtNode>()) {
    Array<ir::Stmt> body = seq_stmt->seq;
    for (int i = 0, n = body.size(); i < n; ++i) {
      f->allow_concise_scoping = (i == n - 1);
      Doc doc = d->AsDoc(body[i], p->Attr("seq")->ArrayIndex(i));
      doc->source_paths.push_back(p);
      if (const auto* block = doc.as<StmtBlockDocNode>()) {
        f->stmts.insert(f->stmts.end(), block->stmts.begin(), block->stmts.end());
      } else {
        f->stmts.push_back(runtime::Downcast<StmtDoc>(doc));
      }
    }
  } else {
    f->allow_concise_scoping = true;
    Doc doc = d->AsDoc(stmt, p);
    if (const auto* block = doc.as<StmtBlockDocNode>()) {
      f->stmts.insert(f->stmts.end(), block->stmts.begin(), block->stmts.end());
    } else {
      f->stmts.push_back(runtime::Downcast<StmtDoc>(doc));
    }
  }
}

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
