// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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
#include <matxscript/ir/prim_var.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;
using namespace ::matxscript::ir::printer;

// PrimVar
PrimVar::PrimVar(StringRef name_hint, DataType dtype, Span span) {
  auto n = make_object<PrimVarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = std::move(dtype);
  n->checked_type_ = PrimType(n->dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

PrimVar::PrimVar(StringRef name_hint, Type type_annotation, Span span) {
  auto n = make_object<PrimVarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = GetRuntimeDataType(type_annotation);
  n->type_annotation = std::move(type_annotation);
  n->checked_type_ = n->type_annotation;
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimVar")
    .set_body_typed([](StringRef name_hint, runtime::RTValue type, Span span) {
      if (type.IsObjectRef<Type>()) {
        return PrimVar(name_hint, type.As<Type>(), span);
      } else {
        return PrimVar(name_hint, type.As<DataType>(), span);
      }
    });

MATXSCRIPT_REGISTER_NODE_TYPE(PrimVarNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrimVarNode*>(node.get());
      // omit the type
      // stream << op->name << "." << op->type;
      p->stream << op->name_hint;
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<PrimVar>("", [](PrimVar var, ObjectPath p, IRDocsifier d) -> Doc {
      return IdDoc(var->name_hint);
    });

// PrimIterVar
PrimIterVar::PrimIterVar(RangeExpr dom, PrimVar var, Span span) {
  ObjectPtr<PrimIterVarNode> n = make_object<PrimIterVarNode>();
  MXCHECK(var.dtype().is_int()) << "Expect var type is 'int' but get '" << var.dtype() << "'";
  n->dom = std::move(dom);
  n->var = std::move(var);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimIterVar")
    .set_body_typed([](RangeExpr dom, PrimVar var, Span span) {
      return PrimIterVar(std::move(dom), std::move(var), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(PrimIterVarNode);

}  // namespace ir
}  // namespace matxscript
