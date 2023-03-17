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
#include <matxscript/ir/none_expr.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace runtime;
using namespace ::matxscript::ir::printer;

NoneExpr::NoneExpr(Span span) {
  ObjectPtr<NoneExprNode> n = make_object<NoneExprNode>();
  n->span = std::move(span);
  n->checked_type_ = ObjectType();
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(NoneExprNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.NoneExpr").set_body_typed([]() {
  static NoneExpr none;
  return none;
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<NoneExprNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "NoneExpr";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<NoneExpr>("", [](NoneExpr t, ObjectPath p, IRDocsifier d) -> Doc {
      return LiteralDoc::None(p);
    });

}  // namespace ir
}  // namespace matxscript
