// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the expressions is inspired by TVM.
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
 * \file src/ir/class_stmt.cc
 * \brief The function data structure.
 */
#include <matxscript/ir/class_stmt.h>

#include <matxscript/ir/_base/with.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/ir/printer/utils.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;
using namespace ::matxscript::ir::printer;

ClassStmt::ClassStmt(
    StringRef name, Stmt base, Array<Stmt> body, ClassType type, DictAttrs attrs, Span span) {
  auto n = make_object<ClassStmtNode>();
  n->name = std::move(name);
  n->base = std::move(base);
  n->body = std::move(body);
  n->type = std::move(type);
  if (attrs.defined()) {
    n->attrs = std::move(attrs);
  } else {
    n->attrs = DictAttrs(Map<StringRef, ObjectRef>{});
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ClassStmtNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassStmt")
    .set_body_typed([](StringRef name,
                       Stmt base,
                       Array<Stmt> body,
                       ClassType type,
                       DictAttrs attrs,
                       Span span) {
      return ClassStmt(std::move(name),
                       std::move(base),
                       std::move(body),
                       std::move(type),
                       std::move(attrs),
                       std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::ClassStmt>("", [](ir::ClassStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, stmt);
      (*f)->AddDispatchToken(d, "ir");
      d->SetCommonPrefix(stmt, [](const ObjectRef& obj) {
        return obj->IsInstance<ir::PrimVarNode>() || obj->IsInstance<ir::BufferNode>();
      });
      // Step 1. Handle `class->base`
      StringRef base = "object";
      if (stmt->base.defined()) {
        base = Downcast<ClassStmt>(stmt->base)->name;
      }
      // Step 2. Handle `class->attrs`
      if (stmt->attrs.defined() && !stmt->attrs->dict.empty()) {
        auto attrs = IRTextPrinter::Print(stmt->attrs, d->cfg);
        (*f)->stmts.push_back(CommentDoc(attrs));
      }
      // Step 3. Handle `class->body`
      // TODO: change first_arg_name from "this" to "self"
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);

      return ClassDoc(
          /*name=*/IdDoc(stmt->name),
          /*base=*/IdDoc(base),
          /*decorators=*/{Dialect(d, "script")},
          /*body=*/(*f)->stmts);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ClassStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << IRTextPrinter::Print(node, NullOpt);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassStmt_Attrs").set_body_typed([](ClassStmt cls) {
  return cls->attrs;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassStmt_Copy").set_body_typed([](ClassStmt stmt) { return stmt; });

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassStmt_WithAttr")
    .set_body_typed([](ClassStmt stmt, StringRef key, RTValue arg_val) -> ClassStmt {
      ObjectRef value = StringRef::CanConvertFrom(arg_val) ? arg_val.As<StringRef>()
                                                           : arg_val.AsObjectRef<ObjectRef>();
      return WithAttr(std::move(stmt), std::move(key), value);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassStmt_GetType").set_body_typed([](ClassStmt stmt) {
  return stmt->type;
});

}  // namespace ir
}  // namespace matxscript
