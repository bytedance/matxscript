// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by Halide/TVM IR.
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
 * \file  module.cc
 * \brief The global module in Relay.
 */
// clang-format off
#include <matxscript/ir/module.h>

#include <fstream>
#include <sstream>
#include <unordered_set>

#include <matxscript/ir/_base/structural_equal.h>
#include <matxscript/ir/_base/with.h>
#include <matxscript/runtime/registry.h>
// NOTE: reverse dependency on relay.
// These dependencies do not happen at the interface-level,
// and are only used in minimum cases where they are clearly marked.
//
// Rationale: We calls into relay's analysis module to verify correctness.
#include <matxscript/ir/type_functor.h>
#include <matxscript/ir/analysis.h>
#include <matxscript/ir/expr_functor.h>
// clang-format on
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/ir/printer/utils.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;
using namespace ::matxscript::ir::printer;

IRModule::IRModule(Array<Stmt> body) {
  auto n = make_object<IRModuleNode>();
  n->body = std::move(body);
  data_ = std::move(n);
}

bool IRModuleNode::SEqualReduce(const IRModuleNode* other, SEqualReducer equal) const {
  if (body.size() != other->body.size())
    return false;
  return equal(body, other->body);
}

void IRModuleNode::SHashReduce(SHashReducer hash_reduce) const {
  hash_reduce(body);
}

void IRModuleNode::AddExportFunction(const StringRef& func_name) {
  // TODO: remove this function
  const auto* BaseFuncWithAttr =
      ::matxscript::runtime::FunctionRegistry::Get("ir.BaseFuncWithAttr");

  auto fn_mutate = [&func_name, &BaseFuncWithAttr](Stmt s) -> Stmt {
    if (const auto* fn_node = s.as<BaseFuncNode>()) {
      if (fn_node->GetGlobalName() == func_name) {
        BaseFunc func =
            (*BaseFuncWithAttr)({s, String(attr::kExportSymbol), Bool(true)}).As<BaseFunc>();
        return func;
      }
    }
    return s;
  };

  auto mod_mutate = [&fn_mutate](Stmt s) -> Stmt {
    if (const auto* cls_node = s.as<ClassStmtNode>()) {
      auto new_body = cls_node->body.Map(fn_mutate);
      if (new_body.same_as(cls_node->body)) {
        return s;
      }
      auto new_cls = Downcast<ClassStmt>(s);
      new_cls.CopyOnWrite()->body = new_body;
      return new_cls;
    } else {
      return fn_mutate(s);
    }
  };

  Array<Stmt> new_body = this->body.Map(mod_mutate);
  this->body = new_body;
}

void IRModuleNode::Add(const Stmt& stmt) {
  this->body.push_back(stmt);
}

void IRModuleNode::Update(const IRModule& mod) {
  for (auto stmt : mod->body) {
    this->Add(stmt);
  }
}

Stmt IRModuleNode::Lookup(const StringRef& name) const {
  for (auto stmt : this->body) {
    if (const auto* fn_node = stmt.as<BaseFuncNode>()) {
      if (fn_node->GetGlobalName() == name) {
        return stmt;
      }
    } else if (const auto* cls_node = stmt.as<ClassStmtNode>()) {
      if (cls_node->name == name) {
        return stmt;
      }
    }
  }
  MXCHECK(false) << "[IRModule] There is no definition of " << name;
  return Stmt{nullptr};
}

MATXSCRIPT_REGISTER_NODE_TYPE(IRModuleNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.IRModule").set_body_typed([](Array<Stmt> body) {
  return IRModule(std::move(body));
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_Add").set_body([](PyArgs args) -> RTValue {
  IRModule mod = args[0].As<IRModule>();
  Stmt val = args[1].As<Stmt>();
  mod->Add(val);
  return mod;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_Update").set_body_typed([](IRModule mod, IRModule from) {
  mod->Update(from);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_Lookup").set_body_typed([](IRModule mod, StringRef name) {
  return mod->Lookup(name);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_AddExportFunction")
    .set_body_typed([](IRModule mod, StringRef export_func) {
      mod->AddExportFunction(export_func);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IRModuleNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IRModuleNode*>(ref.get());
      p->stream << "IRModule(" << node->body << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IRModule>("", [](IRModule mod, ObjectPath p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, ObjectRef{nullptr});
      (*f)->AddDispatchToken(d, "ir");
      for (int i = 0; i < mod->body.size(); ++i) {
        auto stmt = mod->body[i];
        Doc doc = d->AsDoc(stmt, p->Attr("body")->ArrayIndex(i));
        if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
          (*f)->stmts.push_back(stmt_block->stmts.back());
          (*f)->stmts.back()->source_paths = std::move(doc->source_paths);
        } else if (const auto* stmt = doc.as<StmtDocNode>()) {
          (*f)->stmts.push_back(GetRef<StmtDoc>(stmt));
        } else if (const auto* stmt = doc.as<FunctionDocNode>()) {
          (*f)->stmts.push_back(GetRef<FunctionDoc>(stmt));
        } else {
          (*f)->stmts.push_back(Downcast<ClassDoc>(doc));
        }
      }
      return ModuleDoc((*f)->stmts);
    });

}  // namespace ir
}  // namespace matxscript
