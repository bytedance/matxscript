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

/*!
 * \file matx/ir/stmt_functor.h
 *
 * \brief Functors for tir stmts
 *        utility functions to call common functors.
 */
#pragma once

#include <unordered_map>
#include <utility>

#include <matxscript/ir/expr.h>
#include <matxscript/ir/expr_functor.h>
#include <matxscript/ir/stmt.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::NodeFunctor;

/*!
 * \brief Same as ExprFunctor except it is applied on statements
 * \tparam FType The function signature.
 * \sa ExprFunctor
 */
template <typename FType>
class StmtFunctor;

#define STMT_FUNCTOR_DEFAULT \
  { return VisitStmtDefault_(op, std::forward<Args>(args)...); }

#define IR_STMT_FUNCTOR_DISPATCH(OP)                                                       \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitStmt_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class StmtFunctor<R(const Stmt& n, Args... args)> {
 private:
  using TSelf = StmtFunctor<R(const Stmt& n, Args... args)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args... args)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~StmtFunctor() {
  }
  /*!
   * \brief Same as call.
   * \param n The stmt node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Stmt& n, Args... args) {
    return VisitStmt(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The stmt node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitStmt(const Stmt& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitStmt_(const AllocaVarStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AssignStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ReturnStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const LetStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AttrStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const IfThenElseNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ExceptionHandlerNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const TryExceptNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const RaiseNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ForNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AutoForNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AssertStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const SeqStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const EvaluateNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const WhileNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const BreakNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ContinueNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ExprStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const HLOYieldNode* op, Args... args) STMT_FUNCTOR_DEFAULT;

  virtual R VisitStmtDefault_(const Object* op, Args...) {
    MXLOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    IR_STMT_FUNCTOR_DISPATCH(AllocaVarStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(AssignStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(ReturnStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(LetStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(AttrStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(IfThenElseNode);
    IR_STMT_FUNCTOR_DISPATCH(ExceptionHandlerNode);
    IR_STMT_FUNCTOR_DISPATCH(TryExceptNode);
    IR_STMT_FUNCTOR_DISPATCH(RaiseNode);
    IR_STMT_FUNCTOR_DISPATCH(ForNode);
    IR_STMT_FUNCTOR_DISPATCH(AutoForNode);
    IR_STMT_FUNCTOR_DISPATCH(AssertStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(SeqStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(EvaluateNode);
    IR_STMT_FUNCTOR_DISPATCH(WhileNode);
    IR_STMT_FUNCTOR_DISPATCH(BreakNode);
    IR_STMT_FUNCTOR_DISPATCH(ContinueNode);
    IR_STMT_FUNCTOR_DISPATCH(ExprStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(HLOYieldNode);
    return vtable;
  }
};

#undef IR_STMT_FUNCTOR_DISPATCH
#undef STMT_FUNCTOR_DEFAULT

/*!
 * \brief StmtVisitor.
 */
class MATX_DLL StmtVisitor : protected StmtFunctor<void(const Stmt&)> {
 public:
  using StmtFunctor::operator();

 protected:
  using StmtFunctor::VisitStmt;
  virtual void VisitExpr(const BaseExpr& e) = 0;
  virtual void VisitExpr(const PrimExpr& e) = 0;
  virtual void VisitExpr(const HLOExpr& e) = 0;
  // statement visitor
  void VisitStmt_(const AllocaVarStmtNode* op) override;
  void VisitStmt_(const AssignStmtNode* op) override;
  void VisitStmt_(const ReturnStmtNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const ExceptionHandlerNode* op) override;
  void VisitStmt_(const TryExceptNode* op) override;
  void VisitStmt_(const RaiseNode* op) override;
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const AutoForNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
  void VisitStmt_(const WhileNode* op) override;
  void VisitStmt_(const BreakNode* op) override;
  void VisitStmt_(const ContinueNode* op) override;
  void VisitStmt_(const ExprStmtNode* op) override;
  void VisitStmt_(const HLOYieldNode* op) override;
};

/*!
 * \brief StmtMutator that mutates the statements.
 */
class MATX_DLL StmtMutator : protected StmtFunctor<Stmt(const Stmt&)> {
 public:
  /*!
   * \brief Mutate stmt.
   * \param stmt The input statement to be mutated.
   * \return The result of the call
   * \note It is important that stmt is passed by value.
   *       so copy on write can be triggered correctly.
   *       do mutator(std::move(stmt)) or when copy elison is triggered.
   */
  Stmt operator()(const Stmt& stmt) {
    allow_copy_on_write_ = true;
    return VisitStmt(stmt);
  }

  Stmt Mutate(const Stmt& stmt) {
    allow_copy_on_write_ = true;
    return VisitStmt(stmt);
  }

 protected:
  // We perform copy on write optimizations on the StmtMutator
  // so that an unique copy of parent can be mutated inplace
  // when some of its children changed.
  // We only do such optimization for Stmt nests(instead of Exprs) for now
  // as Stmt's parent state is more likely remain unchanged when one of
  // its child block changes.
  /*!
   * \brief Internal state to indicate whether copy on write is enabled.
   *  COW is enabled iff all the parents of the node are unique.
   */
  bool allow_copy_on_write_{false};
  /*!
   * \brief Perform copy on write on node.
   *
   *  If CopyOnWrite is allowed, directly return
   *  a strong reference to the node container.
   *  Otherwise, return a copy of the node.
   *
   * \return The result object pointer.
   */
  template <typename TNode>
  ObjectPtr<TNode> CopyOnWrite(const TNode* node) {
    if (allow_copy_on_write_) {
      // return the old node.
      return runtime::GetObjectPtr<TNode>(const_cast<TNode*>(node));
    } else {
      // Make a new copy of the node.
      // need to rely on the default copy constructor
      return runtime::make_object<TNode>(*node);
    }
  }
  /*!
   * \brief Internal mutator that everyone calls.
   * \note To override mutate's behavior, override VisitExpr instead.
   * \param stmt The input stmt.
   * \return The mutated results.
   */
  Stmt VisitStmt(const Stmt& stmt) override {
    if (allow_copy_on_write_ && !stmt.unique()) {
      allow_copy_on_write_ = false;
      Stmt ret = StmtFunctor::VisitStmt(stmt);
      allow_copy_on_write_ = true;
      return ret;
    } else {
      return StmtFunctor::VisitStmt(stmt);
    }
  }
  virtual BaseExpr VisitExpr(const BaseExpr& e) = 0;
  virtual PrimExpr VisitExpr(const PrimExpr& e) = 0;
  virtual HLOExpr VisitExpr(const HLOExpr& e) = 0;
  // statement visitor
  Stmt VisitStmt_(const AllocaVarStmtNode* op) override;
  Stmt VisitStmt_(const AssignStmtNode* op) override;
  Stmt VisitStmt_(const ReturnStmtNode* op) override;
  Stmt VisitStmt_(const AttrStmtNode* op) override;
  Stmt VisitStmt_(const IfThenElseNode* op) override;
  Stmt VisitStmt_(const ExceptionHandlerNode* op) override;
  Stmt VisitStmt_(const TryExceptNode* op) override;
  Stmt VisitStmt_(const RaiseNode* op) override;
  Stmt VisitStmt_(const LetStmtNode* op) override;
  Stmt VisitStmt_(const ForNode* op) override;
  Stmt VisitStmt_(const AutoForNode* op) override;
  Stmt VisitStmt_(const AssertStmtNode* op) override;
  Stmt VisitStmt_(const SeqStmtNode* op) override;
  Stmt VisitStmt_(const EvaluateNode* op) override;
  Stmt VisitStmt_(const WhileNode* op) override;
  Stmt VisitStmt_(const BreakNode* op) override;
  Stmt VisitStmt_(const ContinueNode* op) override;
  Stmt VisitStmt_(const ExprStmtNode* op) override;
  Stmt VisitStmt_(const HLOYieldNode* op) override;

  /*!
   * \brief Alternative advance method for SeqStmtNode.
   *
   *  This function can be called when a child class override
   *  VisitStmt_(const SeqStmtNode*) to introduce
   *  the special behavior to visit
   *
   * \param op The sequence.
   * \param flatten_before_visit Whether to flatten the sequence before visit.
   * \param fmutate The mutate function, can be nullptr, which defaults to Visit.
   * \return The mutated result.
   */
  Stmt VisitSeqStmt_(const SeqStmtNode* op,
                     bool flatten_before_visit,
                     std::function<Stmt(const Stmt&)> fmutate = nullptr);
  // internal helper.
  class Internal;
};

/*!
 * \brief Visitor that recursively visit stmts and exprs on them.
 */
class StmtExprVisitor : public StmtVisitor, public ExprVisitor {
 public:
  using StmtVisitor::operator();
  using ExprVisitor::operator();

 protected:
  using StmtVisitor::VisitStmt;

  void VisitExpr(const BaseExpr& e) override {
    return ExprVisitor::VisitExpr(e);
  }
  void VisitExpr(const PrimExpr& e) override {
    return ExprVisitor::VisitExpr(e);
  }
  void VisitExpr(const HLOExpr& e) override {
    return ExprVisitor::VisitExpr(e);
  }

  void VisitExpr_(const PrimFuncNode* op) override;
  void VisitExpr_(const FunctionNode* op) override;
  void VisitExpr_(const LambdaFunctionNode* op) override;
};

/*!
 * \brief Mutator that recursively mutates stmts and exprs on them.
 */
class StmtExprMutator : public StmtMutator, public ExprMutator {
 public:
  using StmtMutator::operator();
  using ExprMutator::operator();

 protected:
  using StmtMutator::Mutate;

  BaseFunc VisitExpr(const BaseFunc& f) {
    return runtime::Downcast<BaseFunc>(ExprMutator::VisitExpr(f));
  }

  BaseExpr VisitExpr(const BaseExpr& e) override {
    return ExprMutator::VisitExpr(e);
  }
  PrimExpr VisitExpr(const PrimExpr& e) override {
    return ExprMutator::VisitExpr(e);
  }
  HLOExpr VisitExpr(const HLOExpr& e) override {
    return ExprMutator::VisitExpr(e);
  }

  HLOExpr VisitExpr_(const PrimFuncNode* op) override;
  HLOExpr VisitExpr_(const FunctionNode* op) override;
  HLOExpr VisitExpr_(const LambdaFunctionNode* op) override;
};

/*!
 * \brief recursively visit the ir nodes in post DFS order, and transform it
 *
 * \param stmt The ir to be transformed.
 * \param preorder The function called in before recursive mutation
 *          If preorder returns None, then the transform will proceed to recursive call.
 *          If preorder returns a not None Stmt/Expr, the transformer will simply return it and
 *          won't do further recursion.
 * \param postorder The function called after recursive mutation.
 *          The recursive mutation result is passed to postorder for further mutation.
 * \param only_enable List of runtime::String.
 *          If it is null, all IRNode will call preorder/postorder
 *          If it is not null, preorder/postorder will only be called
 *          when the IRNode's type key is in the list.
 */
MATX_DLL Stmt
IRTransform(Stmt stmt,
            const runtime::NativeFunction& preorder,
            const runtime::NativeFunction& postorder,
            runtime::Optional<runtime::Array<StringRef>> only_enable = runtime::NullOpt);

/*!
 * \brief Recursively visit the ir in post DFS order node, apply fvisit
 * Each node is guaranteed to be visited only once.
 * \param node The ir to be visited.
 * \param fvisit The visitor function to be applied.
 */
MATX_DLL void PostOrderVisit(const ObjectRef& node, std::function<void(const ObjectRef&)> fvisit);

/*!
 * \brief Substitute the var specified by vmap.
 * \param stmt The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 * \return The converted form.
 */
MATX_DLL Stmt Substitute(Stmt stmt,
                         std::function<runtime::Optional<PrimExpr>(const PrimVar& var)> vmap);
MATX_DLL Stmt Substitute(Stmt stmt,
                         std::function<runtime::Optional<HLOExpr>(const HLOVar& var)> vmap);

/*!
 * \brief Substitute the var specified by vmap.
 * \param expr The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 * \return The result.
 */
MATX_DLL PrimExpr Substitute(PrimExpr expr,
                             std::function<runtime::Optional<PrimExpr>(const PrimVar& var)> vmap);

MATX_DLL HLOExpr Substitute(HLOExpr expr,
                            std::function<runtime::Optional<HLOExpr>(const HLOVar& var)> vmap);

MATX_DLL BaseExpr Substitute(BaseExpr expr,
                             std::function<runtime::Optional<BaseExpr>(const BaseExpr& var)> vmap);

/*!
 * \brief Sugar for substitute via a given map.
 * \param input The input to be updated.
 * \param value_map The map of new values.
 * \return The result.
 * \tparam T the input type, can be PrimExpr or Stmt.
 */
template <typename T>
inline auto Substitute(T input, const runtime::Map<PrimVar, PrimExpr>& value_map) {
  auto vmap = [&](const PrimVar& var) -> runtime::Optional<PrimExpr> {
    auto it = value_map.find(var);
    if (it != value_map.end())
      return (*it).second;
    return runtime::Optional<PrimExpr>(nullptr);
  };
  return Substitute(std::move(input), vmap);
}

template <typename T>
inline auto Substitute(T input, const runtime::Map<HLOVar, HLOExpr>& value_map) {
  auto vmap = [&](const HLOVar& var) -> runtime::Optional<HLOExpr> {
    auto it = value_map.find(var);
    if (it != value_map.end())
      return (*it).second;
    return runtime::Optional<HLOExpr>(nullptr);
  };
  return Substitute(std::move(input), vmap);
}

/*!
 * \brief Sugar for substitute via a given map.
 * \param input The input to be updated.
 * \param value_map The map of new values.
 * \return The result.
 * \tparam T the input type, can be PrimExpr or Stmt.
 */
template <typename T>
inline T Substitute(T input, const std::unordered_map<const PrimVarNode*, PrimExpr>& value_map) {
  auto vmap = [&](const PrimVar& var) -> runtime::Optional<PrimExpr> {
    auto it = value_map.find(var.get());
    if (it != value_map.end())
      return (*it).second;
    return runtime::Optional<PrimExpr>(nullptr);
  };
  return Substitute(std::move(input), vmap);
}

template <typename T>
inline T Substitute(T input, const std::unordered_map<const HLOVarNode*, HLOExpr>& value_map) {
  auto vmap = [&](const HLOVar& var) -> runtime::Optional<HLOExpr> {
    auto it = value_map.find(var.get());
    if (it != value_map.end())
      return (*it).second;
    return runtime::Optional<HLOExpr>(nullptr);
  };
  return Substitute(std::move(input), vmap);
}

}  // namespace ir
}  // namespace matxscript
