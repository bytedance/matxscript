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
#include <matxscript/ir/hlo_var.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using runtime::make_object;

MATXSCRIPT_REGISTER_NODE_TYPE(IdNode);

Id::Id(StringRef name_hint) {
  ObjectPtr<IdNode> n = make_object<IdNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

HLOVar::HLOVar(Id vid, Type type_annotation, Span span) {
  ObjectPtr<HLOVarNode> n = make_object<HLOVarNode>();
  n->vid = std::move(vid);
  n->checked_type_ = type_annotation;
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(HLOVarNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOVar")
    .set_body_typed([](StringRef str, Type type_annotation, Span span) {
      return HLOVar(str, type_annotation, span);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HLOVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const HLOVarNode*>(ref.get());
      p->stream << "HLOVar(" << node->name_hint();
      if (node->type_annotation.defined()) {
        p->stream << ", ty=";
        p->Print(node->type_annotation);
      }
      p->stream << ")";
    });

GlobalVar::GlobalVar(StringRef name_hint, Span span) {
  ObjectPtr<GlobalVarNode> n = make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(GlobalVarNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.GlobalVar").set_body_typed([](StringRef name, Span span) {
  return GlobalVar(name, span);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<GlobalVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const GlobalVarNode*>(ref.get());
      p->stream << "GlobalVar(" << node->name_hint << ")";
    });

}  // namespace ir
}  // namespace matxscript
