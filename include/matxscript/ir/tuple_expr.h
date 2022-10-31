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

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/base.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

/*! \brief Tuple container */
class TupleNode : public HLOExprNode {
 public:
  /*! \brief the fields of the tuple */
  runtime::Array<BaseExpr> fields;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const TupleNode* other, SEqualReducer equal) const {
    // specially handle empty tuple as a constant is not a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    if (fields.size() != 0) {
      hash_reduce->MarkGraphNode();
      hash_reduce(fields);
    }
  }

  static constexpr const char* _type_key = "ir.Tuple";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TupleNode, HLOExprNode);
};

class Tuple : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param fields The fields of a tuple.
   * \param span The source span of the expression.
   */
  MATX_DLL explicit Tuple(runtime::Array<BaseExpr> fields, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Tuple, HLOExpr, TupleNode);
};

}  // namespace ir
}  // namespace matxscript
