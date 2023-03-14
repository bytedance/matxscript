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
#include <matxscript/ir/prim_expr.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {
namespace printer {

using runtime::Downcast;

IRTextPrinter::FType& IRTextPrinter::vtable() {
  static FType inst;
  return inst;
}

StringRef IRTextPrinter::Print(const ObjectRef& node, const Optional<PrinterConfig>& cfg) {
  if (!IRTextPrinter::vtable().can_dispatch(node)) {
    return StringRef(node->GetTypeKey());
    // return AsLegacyRepr(node);
  }
  return IRTextPrinter::vtable()(node, cfg.value_or(PrinterConfig()));
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
