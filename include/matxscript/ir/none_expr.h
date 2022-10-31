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
#pragma once

#include <matxscript/ir/base.h>

namespace matxscript {
namespace ir {

class NoneExprNode : public HLOExprNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const NoneExprNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  static constexpr const char* _type_key = "ir.NoneExpr";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(NoneExprNode, HLOExprNode);
};

class NoneExpr : public HLOExpr {
 public:
  /*!
   * \brief The constructor
   * \param span The source span of the expression.
   */
  MATX_DLL explicit NoneExpr(Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(NoneExpr, HLOExpr, NoneExprNode);
};

}  // namespace ir
}  // namespace matxscript
