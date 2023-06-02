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
/*!
 * \file matxscript/ir/_base/repr_printer.h
 * \brief Printer class to print repr string of each AST/IR nodes.
 */
#pragma once

#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/functor.h>

#include <matxscript/ir/_base/cow_array_ref.h>
#include <matxscript/ir/_base/cow_map_ref.h>
#include <matxscript/ir/_base/object_path.h>
#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/ir/printer/printer.h>

#include <iostream>
#include <string>

namespace matxscript {
namespace ir {
namespace printer {

class PrinterConfigNode : public Object {
 public:
  /*! \brief A stack that tracks the names of the binding hierarchy */
  Array<StringRef> binding_names = {};
  /*! \brief The prefix of module */
  StringRef dialect_prefix = "matx";
  /*! \brief Ignore Type Cast */
  bool ignore_type_cast = true;
  /*! \brief Number of spaces used for indentation*/
  int indent_spaces = 4;
  /*! \brief Whether to print line numbers */
  bool print_line_numbers = false;
  /*! \brief Number of context lines to print around the underlined text */
  int num_context_lines = -1;
  /* \brief Object path to be underlined */
  Array<ObjectPath> path_to_underline = Array<ObjectPath>();
  /*! \brief Object path to be annotated. */
  Map<ObjectPath, StringRef> path_to_annotate = {};
  /*! \brief Object to be underlined. */
  Array<ObjectRef> obj_to_underline = Array<ObjectRef>();
  /*! \brief Object to be annotated. */
  Map<ObjectRef, StringRef> obj_to_annotate = {};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("binding_names", &binding_names);
    v->Visit("dialect_prefix", &dialect_prefix);
    v->Visit("indent_spaces", &indent_spaces);
    v->Visit("print_line_numbers", &print_line_numbers);
    v->Visit("num_context_lines", &num_context_lines);
    v->Visit("path_to_underline", &path_to_underline);
    v->Visit("path_to_annotate", &path_to_annotate);
    v->Visit("obj_to_underline", &obj_to_underline);
    v->Visit("obj_to_annotate", &obj_to_annotate);
  }

  static constexpr const char* _type_key = "node.PrinterConfig";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrinterConfigNode, Object);
};

class PrinterConfig : public ObjectRef {
 public:
  explicit PrinterConfig(Map<StringRef, ObjectRef> config_dict = Map<StringRef, ObjectRef>());

  MATXSCRIPT_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterConfig,
                                                           runtime::ObjectRef,
                                                           PrinterConfigNode);
};

/*! \brief Legacy behavior of ReprPrinter. */
class IRTextPrinter {
 public:
  /* Convert the ir to text format */
  static StringRef Print(const ObjectRef& node, const Optional<PrinterConfig>& cfg);
  // Allow registration to be printer.
  using FType = runtime::NodeFunctor<StringRef(const ObjectRef&, const PrinterConfig&)>;
  MATX_DLL static FType& vtable();
};

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
