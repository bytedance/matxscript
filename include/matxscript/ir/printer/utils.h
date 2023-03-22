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
#include <unordered_set>
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
 * \brief Defines a buffer in the IRDocsifier at the given frame,
 * and returns the corresponding IdDoc
 * \param buffer The buffer to define
 * \param frame The frame to define the buffer in
 * \param d The IRDocsifier
 * \return The IdDoc corresponding to the buffer
 */
inline IdDoc DefineBuffer(const ir::Buffer& buffer, const Frame& frame, const IRDocsifier& d) {
  return d->Define(buffer, frame, buffer->name.empty() ? "buffer" : buffer->name);
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

/*!
 * \brief Find the top frame in the stack that could place a var definition
 * \param var The var to be defined
 * \param d The IRDocsifier
 * \return The frame that could place the var definition
 */
inline Optional<Frame> FindLowestVarDef(const ObjectRef& var, const IRDocsifier& d) {
  if (!d->common_prefix.count(var.get())) {
    return NullOpt;
  }
  int n_frames = d->frames.size();
  std::unordered_map<const Object*, const FrameNode*> tir_to_frame;
  const FrameNode* fallback_frame = nullptr;
  tir_to_frame.reserve(n_frames);
  for (int i = n_frames - 1; i >= 0; --i) {
    if (const auto* f = d->frames[i].as<IRFrameNode>()) {
      if (f->tir.defined()) {
        tir_to_frame[f->tir.get()] = f;
      } else if (fallback_frame == nullptr) {
        fallback_frame = f;
      }
    }
  }
  const std::vector<const Object*>& path = d->common_prefix.at(var.get());
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (tir_to_frame.count(*it)) {
      return runtime::GetRef<Frame>(tir_to_frame.at(*it));
    }
  }
  if (fallback_frame != nullptr) {
    return runtime::GetRef<Frame>(fallback_frame);
  }
  return NullOpt;
}

/*!
 * \brief Print the creation of a Var
 * \param var The Var to be printed
 * \param var_p The object path of the Var
 * \param d The IRDocsifier
 * \return The ExprDoc corresponding to the Var creation
 */
ExprDoc PrintVarCreation(const ir::PrimVar& var, const ObjectPath& var_p, const IRDocsifier& d);

/*!
 * \brief Generate a name which not in defined_names
 * \param name_hint The init name
 * \param defined_names The collection of existing names
 * \return The unique name
 */
StringRef GenerateUniqueName(StringRef name_hint,
                             const std::unordered_set<StringRef>& defined_names);

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
