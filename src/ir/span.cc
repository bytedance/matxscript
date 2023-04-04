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
/*!
 * \file span.cc
 * \brief The span data structure.
 */
#include <matxscript/ir/span.h>

#include <algorithm>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::ir::printer;

ObjectPtr<Object> GetSourceNameNode(const StringRef& name) {
  // always return pointer as the reference can change as map re-allocate.
  // or use another level of indirection by creating a unique_ptr
  static std::unordered_map<StringRef, ObjectPtr<SourceNameNode>> source_map;

  auto sn = source_map.find(name);
  if (sn == source_map.end()) {
    ObjectPtr<SourceNameNode> n = runtime::make_object<SourceNameNode>();
    source_map[name] = n;
    n->name = std::move(name);
    return n;
  } else {
    return sn->second;
  }
}

ObjectPtr<Object> GetSourceNameNodeByStr(const runtime::String& name) {
  return GetSourceNameNode(name);
}

SourceName SourceName::Get(const StringRef& name) {
  return SourceName(GetSourceNameNode(name));
}

Span::Span(StringRef file_name, int64_t lineno, StringRef func_name, StringRef source_code) {
  auto node = runtime::make_object<SpanNode>();
  node->file_name = std::move(file_name);
  node->lineno = std::move(lineno);
  node->func_name = std::move(func_name);
  node->source_code = std::move(source_code);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.SourceName").set_body_typed(SourceName::Get);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<SourceName>("", [](SourceName s, ObjectPath p, IRDocsifier d) -> Doc {
      return Dialect(d, "SourceName")->Call({LiteralDoc::Str(s->name, p->Attr("name"))});
    });

MATXSCRIPT_REGISTER_NODE_TYPE(SourceNameNode)
    .set_creator(GetSourceNameNodeByStr)
    .set_repr_bytes([](const Object* n) -> runtime::String {
      return static_cast<const SourceNameNode*>(n)->name;
    });

MATXSCRIPT_REGISTER_NODE_TYPE(SpanNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.Span").set_body_typed(
    [](StringRef file_name, int64_t lineno, StringRef func_name, StringRef source_code) {
      return Span(file_name, lineno, func_name, source_code);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Span>("", [](Span s, ObjectPath p, IRDocsifier d) -> Doc {
      Array<StringRef> keys;
      Array<ExprDoc> values;

      keys.push_back("file_name");
      values.push_back(d->AsDoc<ExprDoc>(s->file_name, p->Attr("file_name")));

      keys.push_back("lineno");
      values.push_back(LiteralDoc::Int(s->lineno, p->Attr("lineno")));

      keys.push_back("func_name");
      values.push_back(d->AsDoc<ExprDoc>(s->func_name, p->Attr("func_name")));

      keys.push_back("source_code");
      values.push_back(d->AsDoc<ExprDoc>(s->source_code, p->Attr("source_code")));

      return Dialect(d, "Span")->Call({}, keys, values);
    });

}  // namespace ir
}  // namespace matxscript
