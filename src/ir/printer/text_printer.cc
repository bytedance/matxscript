// Copyright 2022 ByteDance Ltd. and/or its affiliates.
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
#include <matxscript/ir/printer/text_printer.h>

#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/_base/with.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {
namespace printer {

using runtime::Downcast;
using runtime::GetRef;

extern StringRef DocToPythonScript(Doc doc, const PrinterConfig& cfg);

static std::string Docsify(const ObjectRef& obj,
                           const IRDocsifier& d,
                           const Frame& f,
                           const PrinterConfig& cfg) {
  Doc doc = d->AsDoc(obj, ObjectPath::Root());
  bool move_source_paths = false;
  if (const auto* expr_doc = doc.as<ExprDocNode>()) {
    f->stmts.clear();
    f->stmts.push_back(ExprStmtDoc(GetRef<ExprDoc>(expr_doc)));
  } else if (const auto* stmt_doc = doc.as<StmtDocNode>()) {
    f->stmts.push_back(GetRef<StmtDoc>(stmt_doc));
  } else if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
    for (const StmtDoc& d : stmt_block->stmts) {
      f->stmts.push_back(d);
    }
    move_source_paths = true;
  } else {
    MXLOG(FATAL) << "TypeError: Unexpected doc type: " << doc->GetTypeKey();
  }
  std::ostringstream os;
  if (!d->metadata.empty()) {
    f->stmts.push_back(
        CommentDoc("Metadata omitted. Use show_meta=True in script() method to show it."));
  }
  if (move_source_paths) {
    StmtBlockDoc new_doc(f->stmts);
    new_doc->source_paths = std::move(doc->source_paths);
    os << DocToPythonScript(new_doc, cfg);
  } else {
    os << DocToPythonScript(StmtBlockDoc(f->stmts), cfg);
  }
  return os.str();
}

std::string ReprPrintIR(const ObjectRef& obj, const PrinterConfig& cfg) {
  IRDocsifier d(cfg);
  With<IRFrame> f(d);
  (*f)->AddDispatchToken(d, "ir");
  return Docsify(obj, d, *f, cfg);
}

IRTextPrinter::FType& IRTextPrinter::vtable() {
  static FType inst;
  return inst;
}

StringRef IRTextPrinter::Print(const ObjectRef& node, const Optional<PrinterConfig>& cfg_opt) {
  auto cfg = cfg_opt.value_or(PrinterConfig());
  if (!IRTextPrinter::vtable().can_dispatch(node)) {
    if (IRDocsifier::vtable().can_dispatch("ir", node->type_index())) {
      return StringRef(ReprPrintIR(node, cfg));
    } else {
      return StringRef(node->GetTypeKey());
    }
  }
  return IRTextPrinter::vtable()(node, cfg);
}

PrinterConfig::PrinterConfig(Map<StringRef, ObjectRef> config_dict) {
  runtime::ObjectPtr<PrinterConfigNode> n = runtime::make_object<PrinterConfigNode>();
  if (auto v = config_dict.Get("name")) {
    n->binding_names.push_back(Downcast<StringRef>(v));
  }
  if (auto v = config_dict.Get("ir_prefix")) {
    n->ir_prefix = Downcast<StringRef>(v);
  }
  if (auto v = config_dict.Get("tir_prefix")) {
    n->tir_prefix = Downcast<StringRef>(v);
  }
  if (auto v = config_dict.Get("indent_spaces")) {
    n->indent_spaces = Downcast<IntImm>(v)->value;
  }
  if (auto v = config_dict.Get("print_line_numbers")) {
    n->print_line_numbers = Downcast<IntImm>(v)->value;
  }
  if (auto v = config_dict.Get("num_context_lines")) {
    n->num_context_lines = Downcast<IntImm>(v)->value;
  }
  if (auto v = config_dict.Get("path_to_underline")) {
    n->path_to_underline = Downcast<Optional<Array<ObjectPath>>>(v).value_or(Array<ObjectPath>());
  }
  if (auto v = config_dict.Get("path_to_annotate")) {
    n->path_to_annotate =
        Downcast<Optional<Map<ObjectPath, StringRef>>>(v).value_or(Map<ObjectPath, StringRef>());
  }
  if (auto v = config_dict.Get("obj_to_underline")) {
    n->obj_to_underline = Downcast<Optional<Array<ObjectRef>>>(v).value_or(Array<ObjectRef>());
  }
  if (auto v = config_dict.Get("obj_to_annotate")) {
    n->obj_to_annotate =
        Downcast<Optional<Map<ObjectRef, StringRef>>>(v).value_or(Map<ObjectRef, StringRef>());
  }
  this->data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(PrinterConfigNode);
MATXSCRIPT_REGISTER_GLOBAL("node.PrinterConfig")
    .set_body_typed([](Map<StringRef, ObjectRef> config_dict) {
      return PrinterConfig(config_dict);
    });
MATXSCRIPT_REGISTER_GLOBAL("node.IRTextPrinterScript").set_body_typed(IRTextPrinter::Print);

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
