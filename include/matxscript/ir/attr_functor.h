// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Taken from https://github.com/apache/tvm/blob/v0.7/src/ir/attr_functor.h
 * with fixes applied:
 * - add namespace matx::ir for fix conflict with tvm
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
 * \file attr_functor.h
 * \brief A way to define arbitrary function signature
 *        with dispatch on common attributes.
 *
 * Common attributes include:
 *  - int, float, str constants
 *  - array of attributes
 *  - map of attributes
 */
#pragma once

#include <utility>

#include <matxscript/ir/expr.h>
#include <matxscript/runtime/functor.h>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::ArrayNode;
using ::matxscript::runtime::NodeFunctor;

template <typename FType>
class AttrFunctor;

#define ATTR_FUNCTOR_DEFAULT \
  { return VisitAttrDefault_(op, std::forward<Args>(args)...); }

#define ATTR_FUNCTOR_DISPATCH(OP)                                                          \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitAttr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

// A functor for common attribute information.
template <typename R, typename... Args>
class AttrFunctor<R(const ObjectRef& n, Args...)> {
 private:
  using TSelf = AttrFunctor<R(const ObjectRef& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~AttrFunctor() {
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitAttr(const ObjectRef& n, Args... args) {
    static FType vtable = InitVTable();
    if (vtable.can_dispatch(n)) {
      return vtable(n, this, std::forward<Args>(args)...);
    } else {
      return VisitAttrDefault_(n.get(), std::forward<Args>(args)...);
    }
  }
  virtual R VisitAttrDefault_(const Object* node, Args... args) = 0;
  virtual R VisitAttr_(const ArrayNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const IntImmNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const FloatImmNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const StringImmNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const UnicodeImmNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  // deep comparison of symbolic integer expressions.
  virtual R VisitAttr_(const PrimAddNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimSubNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimMulNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimDivNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimModNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimFloorDivNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimFloorModNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimMinNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimMaxNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimGENode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimGTNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimLTNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimLENode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimEQNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimNENode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimAndNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimOrNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimNotNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimCastNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimCallNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const PrimSelectNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  // TODO(matx4): add more attrs support

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    ATTR_FUNCTOR_DISPATCH(ArrayNode);
    ATTR_FUNCTOR_DISPATCH(IntImmNode);
    ATTR_FUNCTOR_DISPATCH(FloatImmNode);
    ATTR_FUNCTOR_DISPATCH(StringImmNode);
    ATTR_FUNCTOR_DISPATCH(UnicodeImmNode);
    ATTR_FUNCTOR_DISPATCH(PrimAddNode);
    ATTR_FUNCTOR_DISPATCH(PrimSubNode);
    ATTR_FUNCTOR_DISPATCH(PrimMulNode);
    ATTR_FUNCTOR_DISPATCH(PrimDivNode);
    ATTR_FUNCTOR_DISPATCH(PrimModNode);
    ATTR_FUNCTOR_DISPATCH(PrimFloorDivNode);
    ATTR_FUNCTOR_DISPATCH(PrimFloorModNode);
    ATTR_FUNCTOR_DISPATCH(PrimMinNode);
    ATTR_FUNCTOR_DISPATCH(PrimMaxNode);
    ATTR_FUNCTOR_DISPATCH(PrimGENode);
    ATTR_FUNCTOR_DISPATCH(PrimGTNode);
    ATTR_FUNCTOR_DISPATCH(PrimLENode);
    ATTR_FUNCTOR_DISPATCH(PrimLTNode);
    ATTR_FUNCTOR_DISPATCH(PrimEQNode);
    ATTR_FUNCTOR_DISPATCH(PrimNENode);
    ATTR_FUNCTOR_DISPATCH(PrimAndNode);
    ATTR_FUNCTOR_DISPATCH(PrimOrNode);
    ATTR_FUNCTOR_DISPATCH(PrimNotNode);
    ATTR_FUNCTOR_DISPATCH(PrimCastNode);
    ATTR_FUNCTOR_DISPATCH(PrimCallNode);
    ATTR_FUNCTOR_DISPATCH(PrimSelectNode);
    return vtable;
  }
};

}  // namespace ir
}  // namespace matxscript
