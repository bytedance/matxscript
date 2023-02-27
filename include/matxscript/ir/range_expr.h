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
#pragma once

#include <matxscript/ir/base.h>

namespace matxscript {
namespace ir {

/*! \brief range over one dimension */
class RangeExprNode : public HLOExprNode {
 public:
  /*! \brief the start of the node */
  PrimExpr start;
  /*! \brief the stop of range */
  PrimExpr stop;
  /*! \brief the step of range */
  PrimExpr step;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("start", &start);
    v->Visit("stop", &stop);
    v->Visit("stop", &step);
  }

  bool SEqualReduce(const RangeExprNode* other, SEqualReducer equal) const {
    return equal(start, other->start) && equal(stop, other->stop) && equal(step, other->step);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(start);
    hash_reduce(stop);
    hash_reduce(step);
  }

  static constexpr const char* _type_key = "RangeExpr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(RangeExprNode, HLOExprNode);
};

/*! \brief RangeExpr container  */
class RangeExpr : public HLOExpr {
 public:
  /*!
   * \brief constructor by begin and end
   * \param start The begin of the range.
   * \param stop The end of the range.
   * \param step The step of the range.
   * \param span The source code info.
   */
  MATX_DLL RangeExpr(PrimExpr start, PrimExpr stop, PrimExpr step, Span span = Span());

  // declare range.
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(RangeExpr, HLOExpr, RangeExprNode);
};

}  // namespace ir
}  // namespace matxscript
