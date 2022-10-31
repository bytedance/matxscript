// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the ExprFunctor is inspired by TVM.
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
 * \file matx/ir/expr_functor.h
 *
 * \brief Functors for ir expressions.
 */
#pragma once

#include <utility>

#include <matxscript/ir/expr.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/op_expr.h>
#include <matxscript/runtime/demangle.h>
#include <matxscript/runtime/functor.h>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::NodeFunctor;

/*!
 * \brief A dynamical functor that dispatches on in the first Expr argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 *
 *  This helps you to avoid to book-keep return value of Visitor via state,
 *  which can cause bugs easily when state is incorrectly maintained.
 *
 * \code
 *  // A functor that set variable to b. and calculate results.
 *  class MyExprFunctor
 *    : public ir::ExprFunctor<int(const Expr&, int)> {
 *   public:
 *    int VisitExpr_(const Variable* op, int b) final {
 *     return b;
 *    }
 *    int VisitExpr_(const IntImm* op, int b) final {
 *      return op->value;
 *    }
 *    int VisitExpr_(const Add* op, int b) final {
 *     return Visit(op->a, b) + Visit(op->b, b);
 *    }
 *  };
 *  MyExprFunctor f;
 *  Var x("x");
 *  CHECK_EQ(f(x + 1, 2), 3);
 * \endcode
 *
 * \note Why do we need this more powerful Functor:
 *
 *  We often need to implement a transformer tasks.
 *  Say we want to take Expr and transform it to some analysis result,
 *  This easily be done incorrectly using plain Visitor. See IRVisitor's
 *  document for possible error cases.
 *
 * \tparam FType function signiture
 *  This type if only defined for FType with function signiture R(const Expr&, Args...)
 */
template <typename FType>
class PrimExprFunctor;
template <typename FType>
class HLOExprFunctor;

/******************************************************************************
 * PrimExprFunctor
 *****************************************************************************/

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT \
  { return VisitExprDefault_(op, std::forward<Args>(args)...); }

#define IR_EXPR_FUNCTOR_DISPATCH(OP)                                                       \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class PrimExprFunctor<R(const PrimExpr& n, Args...)> {
 private:
  using TSelf = PrimExprFunctor<R(const PrimExpr& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~PrimExprFunctor() {
  }
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const PrimExpr& n, Args... args) {
    return VisitExpr(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const PrimExpr& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  // Arithmetic and logic ops
  virtual R VisitExpr_(const PrimAddNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimSubNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimMulNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimDivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimModNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimFloorDivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimFloorModNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimMinNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimMaxNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimEQNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimNENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimLTNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimLENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimGTNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimGENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimAndNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimOrNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimNotNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimSelectNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;

  // constant
  virtual R VisitExpr_(const IntImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloatImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;

  // var/call/let/cast
  virtual R VisitExpr_(const PrimVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimCallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimLetNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimCastNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOCastPrimNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;

  virtual R VisitExprDefault_(const Object* op, Args...) {
    MXTHROW << "[" << runtime::DemangleType(typeid(*this).name()) << "] Do not have a default for "
            << op->GetTypeKey();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch

    // Arithmetic and logic ops
    IR_EXPR_FUNCTOR_DISPATCH(PrimAddNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimSubNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimMulNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimDivNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimModNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimFloorDivNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimFloorModNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimMinNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimMaxNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimEQNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimNENode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimLTNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimLENode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimGTNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimGENode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimAndNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimOrNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimNotNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimSelectNode);
    // constant
    IR_EXPR_FUNCTOR_DISPATCH(IntImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(FloatImmNode);

    // var/let/call/cast
    IR_EXPR_FUNCTOR_DISPATCH(PrimVarNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimCallNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimLetNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimCastNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOCastPrimNode);

    return vtable;
  }
};

/******************************************************************************
 * HLOExprFunctor
 *****************************************************************************/

template <typename R, typename... Args>
class HLOExprFunctor<R(const HLOExpr& n, Args...)> {
 private:
  using TSelf = HLOExprFunctor<R(const HLOExpr& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~HLOExprFunctor() {
  }
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const HLOExpr& n, Args... args) {
    return VisitExpr(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const HLOExpr& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Arithmetic and logic ops
  virtual R VisitExpr_(const HLOAddNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOSubNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOMulNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOFloorDivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOFloorModNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOEqualNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLONotEqualNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOLessThanNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOLessEqualNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOGreaterThanNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOGreaterEqualNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOAndNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOOrNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLONotNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;

  // var/const/let/call...
  virtual R VisitExpr_(const GlobalVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ConstructorNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOIteratorNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const InitializerListNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const InitializerDictNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FunctionNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimFuncNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LambdaFunctionNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const EnumAttrNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ClassGetItemNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const NoneExprNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const StringImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const UnicodeImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOCastNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOMoveNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOEnumerateNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const HLOZipNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;

  virtual R VisitExprDefault_(const Object* op, Args...) {
    MXTHROW << "[" << runtime::DemangleType(typeid(*this).name()) << "] Do not have a default for "
            << op->GetTypeKey();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    // Arithmetic and logic ops
    IR_EXPR_FUNCTOR_DISPATCH(HLOAddNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOSubNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOMulNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOFloorDivNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOFloorModNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOEqualNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLONotEqualNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOLessThanNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOLessEqualNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOGreaterThanNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOGreaterEqualNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOAndNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOOrNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLONotNode);
    // var/const/let/call...
    IR_EXPR_FUNCTOR_DISPATCH(HLOVarNode);
    IR_EXPR_FUNCTOR_DISPATCH(GlobalVarNode);
    IR_EXPR_FUNCTOR_DISPATCH(TupleNode);
    IR_EXPR_FUNCTOR_DISPATCH(CallNode);
    IR_EXPR_FUNCTOR_DISPATCH(ConstructorNode);
    IR_EXPR_FUNCTOR_DISPATCH(InitializerListNode);
    IR_EXPR_FUNCTOR_DISPATCH(InitializerDictNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOIteratorNode);
    IR_EXPR_FUNCTOR_DISPATCH(FunctionNode);
    IR_EXPR_FUNCTOR_DISPATCH(PrimFuncNode);
    IR_EXPR_FUNCTOR_DISPATCH(LambdaFunctionNode);
    IR_EXPR_FUNCTOR_DISPATCH(OpNode);
    IR_EXPR_FUNCTOR_DISPATCH(EnumAttrNode);
    IR_EXPR_FUNCTOR_DISPATCH(ClassGetItemNode);
    IR_EXPR_FUNCTOR_DISPATCH(NoneExprNode);
    IR_EXPR_FUNCTOR_DISPATCH(StringImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(UnicodeImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOCastNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOMoveNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOEnumerateNode);
    IR_EXPR_FUNCTOR_DISPATCH(HLOZipNode);

    return vtable;
  }
};

#undef IR_EXPR_FUNCTOR_DISPATCH
#undef EXPR_FUNCTOR_DEFAULT

/******************************************************************************
 * ExprVisitor and ExprMutator
 *****************************************************************************/
/*!
 * \brief ExprVisitor
 */
class MATX_DLL ExprVisitor : public PrimExprFunctor<void(const PrimExpr&)>,
                             public HLOExprFunctor<void(const HLOExpr&)> {
 public:
  using PrimExprFunctor<void(const PrimExpr&)>::operator();
  using HLOExprFunctor<void(const HLOExpr&)>::operator();

 public:
  void VisitExpr(const PrimExpr& expr) override {
    if (expr.defined()) {
      return PrimExprFunctor<void(const PrimExpr&)>::VisitExpr(expr);
    }
  }
  void VisitExpr(const HLOExpr& expr) override {
    if (expr.defined()) {
      return HLOExprFunctor<void(const HLOExpr&)>::VisitExpr(expr);
    }
  }
  virtual void VisitExpr(const BaseExpr& expr) {
    if (expr.defined()) {
      if (expr->IsInstance<PrimExprNode>()) {
        VisitExpr(runtime::Downcast<PrimExpr>(expr));
      } else if (expr->IsInstance<HLOExprNode>()) {
        VisitExpr(runtime::Downcast<HLOExpr>(expr));
      } else {
        MXTHROW << "[ExprVisitor] not supported expr node: " << expr;
      }
    }
  }
  virtual void operator()(const BaseExpr& expr) {
    return this->VisitExpr(expr);
  }
  // other info
  virtual void VisitType(const Type& t);
  virtual void VisitSpan(const Span& span);

 protected:
  // list of functions to override.
  void VisitExpr_(const PrimAddNode* op) override;
  void VisitExpr_(const PrimSubNode* op) override;
  void VisitExpr_(const PrimMulNode* op) override;
  void VisitExpr_(const PrimDivNode* op) override;
  void VisitExpr_(const PrimModNode* op) override;
  void VisitExpr_(const PrimFloorDivNode* op) override;
  void VisitExpr_(const PrimFloorModNode* op) override;
  void VisitExpr_(const PrimMinNode* op) override;
  void VisitExpr_(const PrimMaxNode* op) override;
  void VisitExpr_(const PrimEQNode* op) override;
  void VisitExpr_(const PrimNENode* op) override;
  void VisitExpr_(const PrimLTNode* op) override;
  void VisitExpr_(const PrimLENode* op) override;
  void VisitExpr_(const PrimGTNode* op) override;
  void VisitExpr_(const PrimGENode* op) override;
  void VisitExpr_(const PrimAndNode* op) override;
  void VisitExpr_(const PrimOrNode* op) override;
  void VisitExpr_(const PrimNotNode* op) override;
  void VisitExpr_(const PrimSelectNode* op) override;

  void VisitExpr_(const IntImmNode* op) override;
  void VisitExpr_(const FloatImmNode* op) override;
  void VisitExpr_(const StringImmNode* op) override;
  void VisitExpr_(const UnicodeImmNode* op) override;

  void VisitExpr_(const PrimVarNode* op) override;
  void VisitExpr_(const PrimLetNode* op) override;
  void VisitExpr_(const PrimCallNode* op) override;
  void VisitExpr_(const PrimCastNode* op) override;
  void VisitExpr_(const HLOCastPrimNode* op) override;

  // HLO expr
  void VisitExpr_(const HLOAddNode* op) override;
  void VisitExpr_(const HLOSubNode* op) override;
  void VisitExpr_(const HLOMulNode* op) override;
  void VisitExpr_(const HLOFloorDivNode* op) override;
  void VisitExpr_(const HLOFloorModNode* op) override;
  void VisitExpr_(const HLOEqualNode* op) override;
  void VisitExpr_(const HLONotEqualNode* op) override;
  void VisitExpr_(const HLOLessThanNode* op) override;
  void VisitExpr_(const HLOLessEqualNode* op) override;
  void VisitExpr_(const HLOGreaterThanNode* op) override;
  void VisitExpr_(const HLOGreaterEqualNode* op) override;
  void VisitExpr_(const HLOAndNode* op) override;
  void VisitExpr_(const HLOOrNode* op) override;
  void VisitExpr_(const HLONotNode* op) override;

  void VisitExpr_(const HLOVarNode* op) override;
  void VisitExpr_(const GlobalVarNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const ConstructorNode* op) override;
  void VisitExpr_(const InitializerListNode* op) override;
  void VisitExpr_(const InitializerDictNode* op) override;
  void VisitExpr_(const HLOIteratorNode* op) override;
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const EnumAttrNode* op) override;
  void VisitExpr_(const ClassGetItemNode* op) override;
  void VisitExpr_(const NoneExprNode* op) override;
  void VisitExpr_(const HLOCastNode* op) override;
  void VisitExpr_(const HLOMoveNode* op) override;
  void VisitExpr_(const HLOEnumerateNode* op) override;
  void VisitExpr_(const HLOZipNode* op) override;
};

/*!
 * \brief ExprMutator that mutates expressions.
 */
class MATX_DLL ExprMutator : public PrimExprFunctor<PrimExpr(const PrimExpr&)>,
                             public HLOExprFunctor<HLOExpr(const HLOExpr&)> {
 public:
  using PrimExprFunctor<PrimExpr(const PrimExpr&)>::operator();
  using HLOExprFunctor<HLOExpr(const HLOExpr&)>::operator();

 public:
  virtual BaseExpr operator()(const BaseExpr& expr) {
    return this->VisitExpr(expr);
  }

  PrimExpr VisitExpr(const PrimExpr& expr) override {
    if (expr.defined()) {
      return PrimExprFunctor<PrimExpr(const PrimExpr&)>::VisitExpr(expr);
    }
    return expr;
  }

  HLOExpr VisitExpr(const HLOExpr& expr) override {
    if (expr.defined()) {
      return HLOExprFunctor<HLOExpr(const HLOExpr&)>::VisitExpr(expr);
    }
    return expr;
  }

  virtual BaseExpr VisitExpr(const BaseExpr& expr) {
    if (expr.defined()) {
      if (expr->IsInstance<PrimExprNode>()) {
        return this->VisitExpr(runtime::Downcast<PrimExpr>(expr));
      } else if (expr->IsInstance<HLOExprNode>()) {
        return this->VisitExpr(runtime::Downcast<HLOExpr>(expr));
      } else {
        MXTHROW << "[ExprMutator] not supported expr node: " << expr;
      }
    }
    return expr;
  }

 protected:
  // list of functions to override.
  PrimExpr VisitExpr_(const PrimAddNode* op) override;
  PrimExpr VisitExpr_(const PrimSubNode* op) override;
  PrimExpr VisitExpr_(const PrimMulNode* op) override;
  PrimExpr VisitExpr_(const PrimDivNode* op) override;
  PrimExpr VisitExpr_(const PrimModNode* op) override;
  PrimExpr VisitExpr_(const PrimFloorDivNode* op) override;
  PrimExpr VisitExpr_(const PrimFloorModNode* op) override;
  PrimExpr VisitExpr_(const PrimMinNode* op) override;
  PrimExpr VisitExpr_(const PrimMaxNode* op) override;
  PrimExpr VisitExpr_(const PrimEQNode* op) override;
  PrimExpr VisitExpr_(const PrimNENode* op) override;
  PrimExpr VisitExpr_(const PrimLTNode* op) override;
  PrimExpr VisitExpr_(const PrimLENode* op) override;
  PrimExpr VisitExpr_(const PrimGTNode* op) override;
  PrimExpr VisitExpr_(const PrimGENode* op) override;
  PrimExpr VisitExpr_(const PrimAndNode* op) override;
  PrimExpr VisitExpr_(const PrimOrNode* op) override;
  PrimExpr VisitExpr_(const PrimNotNode* op) override;
  PrimExpr VisitExpr_(const PrimSelectNode* op) override;
  PrimExpr VisitExpr_(const IntImmNode* op) override;
  PrimExpr VisitExpr_(const FloatImmNode* op) override;

  PrimExpr VisitExpr_(const PrimVarNode* op) override;
  PrimExpr VisitExpr_(const PrimLetNode* op) override;
  PrimExpr VisitExpr_(const PrimCallNode* op) override;
  PrimExpr VisitExpr_(const PrimCastNode* op) override;
  PrimExpr VisitExpr_(const HLOCastPrimNode* op) override;

  // HLO expr
  HLOExpr VisitExpr_(const HLOAddNode* op) override;
  HLOExpr VisitExpr_(const HLOSubNode* op) override;
  HLOExpr VisitExpr_(const HLOMulNode* op) override;
  HLOExpr VisitExpr_(const HLOFloorDivNode* op) override;
  HLOExpr VisitExpr_(const HLOFloorModNode* op) override;
  HLOExpr VisitExpr_(const HLOEqualNode* op) override;
  HLOExpr VisitExpr_(const HLONotEqualNode* op) override;
  HLOExpr VisitExpr_(const HLOLessThanNode* op) override;
  HLOExpr VisitExpr_(const HLOLessEqualNode* op) override;
  HLOExpr VisitExpr_(const HLOGreaterThanNode* op) override;
  HLOExpr VisitExpr_(const HLOGreaterEqualNode* op) override;
  HLOExpr VisitExpr_(const HLOAndNode* op) override;
  HLOExpr VisitExpr_(const HLOOrNode* op) override;
  HLOExpr VisitExpr_(const HLONotNode* op) override;

  HLOExpr VisitExpr_(const HLOVarNode* op) override;
  HLOExpr VisitExpr_(const GlobalVarNode* op) override;
  HLOExpr VisitExpr_(const TupleNode* op) override;
  HLOExpr VisitExpr_(const CallNode* op) override;
  HLOExpr VisitExpr_(const ConstructorNode* op) override;
  HLOExpr VisitExpr_(const InitializerListNode* op) override;
  HLOExpr VisitExpr_(const InitializerDictNode* op) override;
  HLOExpr VisitExpr_(const HLOIteratorNode* op) override;
  HLOExpr VisitExpr_(const OpNode* op) override;
  HLOExpr VisitExpr_(const EnumAttrNode* op) override;
  HLOExpr VisitExpr_(const ClassGetItemNode* op) override;
  HLOExpr VisitExpr_(const NoneExprNode* op) override;
  HLOExpr VisitExpr_(const StringImmNode* op) override;
  HLOExpr VisitExpr_(const UnicodeImmNode* op) override;
  HLOExpr VisitExpr_(const HLOCastNode* op) override;
  HLOExpr VisitExpr_(const HLOMoveNode* op) override;
  HLOExpr VisitExpr_(const HLOEnumerateNode* op) override;
  HLOExpr VisitExpr_(const HLOZipNode* op) override;

 public:
  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);
};

}  // namespace ir
}  // namespace matxscript
