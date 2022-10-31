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
#pragma once

#include <string>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;
using runtime::StringRef;

/*!
 * \brief The source name in the Span
 * \sa SourceNameNode, Span
 */
class SourceName;
/*!
 * \brief The name of a source fragment.
 */
class SourceNameNode : public Object {
 public:
  /*! \brief The source name. */
  runtime::StringRef name;
  // override attr visitor
  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("name", &name);
  }

  bool SEqualReduce(const SourceNameNode* other, runtime::SEqualReducer equal) const {
    return equal(name, other->name);
  }

  static constexpr const char* _type_key = "SourceName";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SourceNameNode, Object);
};

/*!
 * \brief The source name of a file span.
 * \sa SourceNameNode, Span
 */
class SourceName : public ObjectRef {
 public:
  /*!
   * \brief Get an SourceName for a given operator name.
   *  Will raise an error if the source name has not been registered.
   * \param name Name of the operator.
   * \return SourceName valid throughout program lifetime.
   */
  MATX_DLL static SourceName Get(const StringRef& name);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(SourceName, ObjectRef, SourceNameNode);
};

/*!
 * \brief Span information for debugging purposes
 */
class Span;
/*!
 * \brief Stores locations in frontend source that generated a node.
 */
class SpanNode : public Object {
 public:
  /*! \brief The source file name. */
  StringRef file_name;
  /*! \brief The source line number. */
  int64_t lineno;
  /*! \brief The source func name. */
  StringRef func_name;
  /*! \brief The source code line. */
  StringRef source_code;

  // override attr visitor
  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("file_name", &file_name);
    v->Visit("lineno", &lineno);
    v->Visit("func_name", &func_name);
    v->Visit("source_code", &source_code);
  }

  bool SEqualReduce(const SpanNode* other, runtime::SEqualReducer equal) const {
    return equal(file_name, other->file_name) && equal(lineno, other->lineno) &&
           equal(func_name, other->func_name) && equal(source_code, other->source_code);
  }

  void SHashReduce(runtime::SHashReducer hash_reduce) const {
    hash_reduce(file_name);
    hash_reduce(lineno);
    hash_reduce(func_name);
    hash_reduce(source_code);
  }

  static constexpr const char* _type_key = "Span";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SpanNode, Object);
};

class Span : public ObjectRef {
 public:
  MATX_DLL Span(StringRef file_name, int64_t lineno, StringRef func_name, StringRef source_code);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Span, ObjectRef, SpanNode);
};

}  // namespace ir
}  // namespace matxscript
