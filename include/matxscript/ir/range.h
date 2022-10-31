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

#include <matxscript/ir/prim_expr.h>

namespace matxscript {
namespace ir {

/*! \brief range over one dimension */
class RangeNode : public Object {
 public:
  /*! \brief beginning of the node */
  PrimExpr min;
  /*! \brief the extend of range */
  PrimExpr extent;
  /*! \brief constructor */
  RangeNode() {
  }
  RangeNode(PrimExpr min, PrimExpr extent) : min(min), extent(extent) {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("min", &min);
    v->Visit("extent", &extent);
  }

  bool SEqualReduce(const RangeNode* other, SEqualReducer equal) const {
    return equal(min, other->min) && equal(extent, other->extent);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(min);
    hash_reduce(extent);
  }

  static constexpr const char* _type_key = "Range";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(RangeNode, Object);
};

/*! \brief Range constainer  */
class Range : public ObjectRef {
 public:
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   */
  MATX_DLL Range(PrimExpr begin, PrimExpr end);
  /*!
   * \brief construct a new range with min and extent
   *  The corresponding constructor is removed,
   *  because that is counter convention of tradition meaning
   *  of range(begin, end)
   *
   * \param min The minimum range.
   * \param extent The extent of the range.
   */
  static Range FromMinExtent(PrimExpr min, PrimExpr extent);
  // declare range.
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Range, ObjectRef, RangeNode);
};

}  // namespace ir
}  // namespace matxscript
