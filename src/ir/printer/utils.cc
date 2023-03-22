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
#include <matxscript/ir/printer/utils.h>

#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/ir/printer/text_printer.h>
#include <matxscript/runtime/str_escape.h>

namespace matxscript {
namespace ir {
namespace printer {

using runtime::Downcast;
using runtime::GetRef;

static inline StringRef DType2Str(const DataType& dtype) {
  return dtype.is_void() ? "void" : runtime::DLDataType2String(dtype);
}

ExprDoc PrintVarCreation(const ir::PrimVar& var, const ObjectPath& var_p, const IRDocsifier& d) {
  Type type = var->type_annotation;
  ObjectPath type_p = var_p->Attr("type_annotation");
  ExprDoc rhs{nullptr};
  Array<StringRef> kwargs_keys;
  Array<ExprDoc> kwargs_values;

  if (const auto* ptr_type = type.as<PointerTypeNode>()) {
    const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>();
    MXCHECK(prim_type);
    ExprDoc element_type =
        LiteralDoc::DataType(prim_type->dtype, type_p->Attr("element_type")->Attr("dtype"));
    rhs = Dialect(d, "handle");
    rhs->source_paths.push_back(var_p->Attr("dtype"));
    rhs = rhs->Call({element_type}, kwargs_keys, kwargs_values);
  } else {
    rhs = Dialect(d, DType2Str(var->dtype));
    rhs->source_paths.push_back(var_p->Attr("dtype"));
    rhs = rhs->Call({}, kwargs_keys, kwargs_values);
  }
  rhs->source_paths.push_back(type_p);
  return rhs;
}

Doc PrintVar(const ir::PrimVar& var, const ObjectPath& var_p, const IRDocsifier& d) {
  if (!d->IsVarDefined(var)) {
    if (Optional<Frame> opt_f = FindLowestVarDef(var, d)) {
      ExprDoc lhs = DefineVar(var, opt_f.value(), d);
      ExprDoc rhs = PrintVarCreation(var, var_p, d);
      opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
    } else {
      MXLOG(WARNING) << "Didn't find variable definition for: " << var->name_hint;
    }
  }
  if (Optional<ExprDoc> doc = d->GetVarDoc(var)) {
    return doc.value();
  }
  MXLOG(FATAL) << "IndexError: Variable is not defined in the environment: " << var->name_hint;
  return ExprDoc{nullptr};
}

StringRef GenerateUniqueName(StringRef name_hint,
                             const std::unordered_set<StringRef>& defined_names) {
  for (char& c : name_hint) {
    if (c != '_' && !std::isalnum(c)) {
      c = '_';
    }
  }
  StringRef name = name_hint;
  for (int i = 1; defined_names.count(name) > 0; ++i) {
    name = name_hint + "_" + std::to_string(i);
  }
  return name;
}

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
