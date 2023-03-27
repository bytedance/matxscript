// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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
 * \file text_printer.cc
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */
#include "text_printer.h"

#include <string>

#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

static const char* kSemVer = "0.0.5";

printer::Doc TextPrinter::PrintMod(const ir::IRModule& mod) {
  Doc doc;
  int counter = 0;
  // functions
  for (const auto& stmt : mod->body) {
    if (counter++ != 0) {
      doc << Doc::NewLine();
    }
    if (auto* fn_node = stmt.as<ir::BaseFuncNode>()) {
      std::ostringstream os;
      os << "def @" << fn_node->GetGlobalName() << " ";
      doc << ir_text_printer_.PrintFunc(Doc::Text(os.str()), GetRef<ir::BaseFunc>(fn_node));
      doc << Doc::NewLine();
    } else if (auto* cls_node = stmt.as<ir::ClassStmtNode>()) {
      for (const auto& cls_stmt : cls_node->body) {
        if (counter++ != 0) {
          doc << Doc::NewLine();
        }
        auto* fn_node = cls_stmt.as<ir::BaseFuncNode>();
        MXCHECK(fn_node);
        std::ostringstream os;
        os << "def @" << cls_node->name << "." << fn_node->GetBoundName() << " ";
        doc << ir_text_printer_.PrintFunc(Doc::Text(os.str()), GetRef<ir::BaseFunc>(fn_node));
        doc << Doc::NewLine();
      }
    }
  }
  return doc;
}

static String PrettyPrint(const ObjectRef& node) {
  printer::Doc doc;
  doc << TextPrinter(nullptr, false).PrintFinal(node);
  return doc.str();
}

static String AsText(const ObjectRef& node) {
  printer::Doc doc;
  doc << "#[version = \"" << kSemVer << "\"]" << printer::Doc::NewLine();
  runtime::TypedNativeFunction<String(ObjectRef)> ftyped = nullptr;
  doc << TextPrinter(ftyped).PrintFinal(node);
  return doc.str();
}

MATXSCRIPT_REGISTER_GLOBAL("ir.PrettyPrint").set_body_typed(PrettyPrint);

MATXSCRIPT_REGISTER_GLOBAL("ir.AsText").set_body_typed(AsText);

}  // namespace runtime
}  // namespace matxscript
