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

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/base.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace ir {

/*!
 * \brief A variable node in the IR.
 *
 * A variable is uniquely identified by its address.
 *
 * Each variable is only binded once in the following nodes:
 * - Allocate
 * - For
 * - Let
 * - LetStmt
 */
class PrimVarNode : public PrimExprNode {
 public:
  /*!
   * \brief The hint to the variable name.
   * \note Each variable is uniquely identified by its address.
   */
  StringRef name_hint;
  /*!
   * \brief type annotaion of the variable.
   *
   * It is an optional field that provides a refined type of the variable than dtype.
   *
   * \sa matxscript/ir/type.h for discussion of relations between runtime::DataType and Type.
   */
  Type type_annotation;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("name", &name_hint);
    v->Visit("type_annotation", &type_annotation);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimVarNode* other, SEqualReducer equal) const {
    if (!equal(dtype, other->dtype))
      return false;
    if (!equal(type_annotation, other->type_annotation))
      return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(type_annotation);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "ir.PrimVar";
  static constexpr const uint32_t _type_child_slots = 1;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(PrimVarNode, PrimExprNode);
};

/*! \brief a named variable in TIR */
class PrimVar : public PrimExpr {
 public:
  explicit PrimVar(ObjectPtr<Object> n) : PrimExpr(n) {
  }
  /*!
   * \brief Constructor
   * \param name_hint variable name
   * \param dtype data type
   */
  MATX_DLL explicit PrimVar(StringRef name_hint = "v",
                            runtime::DataType dtype = runtime::DataType::Int(32),
                            Span span = Span());
  /*!
   * \brief Constructor which provides a more detailed type annotation.
   * \param name_hint variable name.
   * \param type_annotation The type annotation.
   */
  MATX_DLL explicit PrimVar(StringRef name_hint, Type type_annotation, Span span = Span());

  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const PrimVarNode* operator->() const {
    return get();
  }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const PrimVarNode* get() const {
    return static_cast<const PrimVarNode*>(data_.get());
  }
  /*! \brief type indicate the container type */
  using ContainerType = PrimVarNode;
};

}  // namespace ir
}  // namespace matxscript
