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
 * \file tvm/ir.stmt.h
 * \brief ir.statements.
 */
// Acknowledgement: Many low-level stmts originate from Halide.
#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <matxscript/ir/_base/for_each_fwd.h>
#include <matxscript/ir/base.h>
#include <matxscript/ir/prim_var.h>

namespace matxscript {
namespace ir {
/*!
 * \brief just an expression.
 *
 */
class ExprStmtNode : public StmtNode {
 public:
  /*! \brief The expression. */
  BaseExpr expr;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("expr", &expr);
  }

  bool SEqualReduce(const ExprStmtNode* other, SEqualReducer equal) const {
    return equal(expr, other->expr);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(expr);
  }

  static constexpr const char* _type_key = "ir.ExprStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ExprStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to ExprStmtNode.
 * \sa ExprStmtNode
 */
class ExprStmt : public Stmt {
 public:
  MATX_DLL explicit ExprStmt(BaseExpr expr, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ExprStmt, Stmt, ExprStmtNode);
};

/*!
 * \brief Define an new Var.
 *
 */
class AllocaVarStmtNode : public StmtNode {
 public:
  /*! \brief The var expression. */
  BaseExpr var;
  /*! \brief The init value. */
  BaseExpr init_value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("init_value", &init_value);
  }

  bool SEqualReduce(const AllocaVarStmtNode* other, SEqualReducer equal) const {
    return equal(var, other->var) && equal(init_value, other->init_value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(var);
    hash_reduce(init_value);
  }

  static constexpr const char* _type_key = "ir.AllocaVarStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AllocaVarStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AssignStmtNode.
 * \sa AllocaVarStmtNode
 */
class AllocaVarStmt : public Stmt {
 public:
  MATX_DLL explicit AllocaVarStmt(StringRef name,
                                  Type ty,
                                  BaseExpr init_value = BaseExpr(),
                                  Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(AllocaVarStmt, Stmt, AllocaVarStmtNode);
};

/*!
 * \brief Assign an rhs to lhs.
 *
 */
class AssignStmtNode : public StmtNode {
 public:
  /*! \brief The lhs expression. */
  BaseExpr lhs;
  /*! \brief The rhs expression. */
  BaseExpr rhs;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
  }

  bool SEqualReduce(const AssignStmtNode* other, SEqualReducer equal) const {
    return equal(lhs, other->lhs) && equal(rhs, other->rhs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(lhs);
    hash_reduce(rhs);
  }

  static constexpr const char* _type_key = "ir.AssignStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AssignStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AssignStmtNode.
 * \sa AssignStmtNode
 */
class AssignStmt : public Stmt {
 public:
  MATX_DLL explicit AssignStmt(BaseExpr lhs, BaseExpr rhs, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(AssignStmt, Stmt, AssignStmtNode);
};

/*!
 * \brief Return an expression.
 *  This is mostly used for putting a Return node into Function.
 *
 */
class ReturnStmtNode : public StmtNode {
 public:
  /*! \brief The expression to be returned. */
  BaseExpr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("value", &value);
  }

  bool SEqualReduce(const ReturnStmtNode* other, SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.ReturnStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ReturnStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to ReturnNode.
 * \sa ReturnNode
 */
class ReturnStmt : public Stmt {
 public:
  MATX_DLL explicit ReturnStmt(BaseExpr value, Span span = Span());

  explicit ReturnStmt(int value, Span span = Span()) : ReturnStmt(PrimExpr(value), span) {
  }
  explicit ReturnStmt(float value, Span span = Span()) : ReturnStmt(PrimExpr(value), span) {
  }

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ReturnStmt, Stmt, ReturnStmtNode);
};

/*!
 * \brief Let binding, bind var to value, then run body.
 */
class LetStmtNode : public StmtNode {
 public:
  /*! \brief The variable. */
  PrimVar var;
  /*! \brief The value to be binded. */
  PrimExpr value;
  /*! \brief The body block. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const LetStmtNode* other, SEqualReducer equal) const {
    return equal.DefEqual(var, other->var) && equal(value, other->value) &&
           equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(var);
    hash_reduce(value);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "ir.LetStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(LetStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to LetStmtNode.
 * \sa LetStmtNode
 */
class LetStmt : public Stmt {
 public:
  MATX_DLL LetStmt(PrimVar var, PrimExpr value, Stmt body, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(LetStmt, Stmt, LetStmtNode);
};

/*!
 * \brief Define certain auxiliary attribute for the body to be a symbolic value.
 *  This provide auxiliary information for IR passes that transforms body.
 *
 *  In terms of effect, this is equivalent to Block(Evaluate(value), body).
 *
 *  Examples of possible usage:
 *    - Bound of function, variables.
 *    - Hint which block corresponds to a parallel region.
 */
class AttrStmtNode : public StmtNode {
 public:
  /*! \brief this is attribute about certain node */
  ObjectRef node;
  /*! \brief the type key of the attribute */
  StringRef attr_key;
  /*! \brief The attribute value, value is well defined at current scope. */
  BaseExpr value;
  /*! \brief The body statement to be executed */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("node", &node);
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const AttrStmtNode* other, SEqualReducer equal) const {
    return equal(node, other->node) && equal(attr_key, other->attr_key) &&
           equal(value, other->value) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(node);
    hash_reduce(attr_key);
    hash_reduce(value);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "ir.AttrStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AttrStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AttrStmtNode.
 * \sa AttrStmtNode
 */
class AttrStmt : public Stmt {
 public:
  MATX_DLL AttrStmt(
      ObjectRef node, StringRef attr_key, BaseExpr value, Stmt body, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(AttrStmt, Stmt, AttrStmtNode);
};

/*!
 * \brief Assert condition, if an error occurs, return the error message.
 */
class AssertStmtNode : public StmtNode {
 public:
  /*! \brief Condition to be checked. */
  BaseExpr condition;
  /*! \brief Error message when assertion failed. */
  BaseExpr message;
  /*!
   * \brief Body which this assertion holds true.
   *  Will be executed after the assertion.
   */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("condition", &condition);
    v->Visit("message", &message);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const AssertStmtNode* other, SEqualReducer equal) const {
    return equal(condition, other->condition) && equal(message, other->message) &&
           equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(condition);
    hash_reduce(message);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "ir.AssertStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AssertStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AssertStmtNode.
 * \sa AssertStmtNode
 */
class AssertStmt : public Stmt {
 public:
  MATX_DLL AssertStmt(BaseExpr condition, BaseExpr message, Stmt body, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(AssertStmt, Stmt, AssertStmtNode);
};

/*!
 * \brief The container of seq statement.
 *        Represent a sequence of statements.
 */
class SeqStmtNode : public StmtNode {
 public:
  /*! \brief internal sequence content. */
  runtime::Array<Stmt> seq;

  /*! \return get the size of the sequence */
  size_t size() const {
    return seq.size();
  }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const {
    return seq[index];
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("seq", &seq);
  }

  bool SEqualReduce(const SeqStmtNode* other, SEqualReducer equal) const {
    return equal(seq, other->seq);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(seq);
  }

  static constexpr const char* _type_key = "ir.SeqStmt";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SeqStmtNode, StmtNode);
};

/*! \brief Sequence statement. */
class SeqStmt : public Stmt {
 public:
  /*!
   * \brief Construct SeqStmt.
   * \param seq The sequence.
   */
  MATX_DLL explicit SeqStmt(runtime::Array<Stmt> seq, Span span = Span());

  /*! \return get the size of the sequence */
  size_t size() const {
    return operator->()->size();
  }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const {
    return (*(operator->()))[index];
  }
  /*!
   * \brief Construct a sequence statement by flattening
   *        all the arrays and sequences in the arguments
   *        recursively.
   *
   * - When an argument is nullptr, it will be ignored.
   * - When an argument is an array or a SeqStmt, it will be flattened recursively.
   * - A normal Stmt will be appended to the end of the sequence.
   *
   * \note This function can directly return an element
   *       if it is the only element in the sequence.
   *
   * \param seq_args The list of arguments to be flattened.
   * \tparam Args arguments
   * \return The constructed statement
   */
  template <typename... Args>
  static Stmt Flatten(Args&&... seq_args) {
    runtime::Array<Stmt> seq;
    runtime::detail::for_each(Flattener(&seq), std::forward<Args>(seq_args)...);
    if (seq.size() == 1)
      return seq[0];
    return SeqStmt(seq);
  }
  /*! \brief Helper class to flatten sequence of arguments into Array. */
  class Flattener {
   public:
    explicit Flattener(runtime::Array<Stmt>* seq) : seq_(seq) {
    }

    void operator()(size_t i, const Stmt& stmt) const {
      if (!stmt.defined())
        return;
      if (auto* op = stmt.as<SeqStmtNode>()) {
        operator()(0, op->seq);
      } else {
        seq_->push_back(stmt);
      }
    }

    template <typename T>
    void operator()(size_t i, const T& seq) const {
      for (auto v : seq) {
        this->operator()(0, v);
      }
    }

   private:
    runtime::Array<Stmt>* seq_;
  };

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(SeqStmt, Stmt, SeqStmtNode);
};

/*!
 * \brief IfThenElse statment.
 */
class IfThenElseNode : public StmtNode {
 public:
  /*! \brief The condition. */
  BaseExpr condition;
  /*! \brief The branch to be executed when condition is true. */
  Stmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  Stmt else_case;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("condition", &condition);
    v->Visit("then_case", &then_case);
    v->Visit("else_case", &else_case);
  }

  bool SEqualReduce(const IfThenElseNode* other, SEqualReducer equal) const {
    return equal(condition, other->condition) && equal(then_case, other->then_case) &&
           equal(else_case, other->else_case);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(condition);
    hash_reduce(then_case);
    hash_reduce(else_case);
  }

  static constexpr const char* _type_key = "ir.IfThenElse";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IfThenElseNode, StmtNode);
};

/*!
 * \brief Managed reference to IfThenElseNode.
 * \sa IfThenElseNode
 */
class IfThenElse : public Stmt {
 public:
  MATX_DLL IfThenElse(BaseExpr condition,
                      Stmt then_case,
                      Stmt else_case = Stmt(),
                      Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(IfThenElse, Stmt, IfThenElseNode);
};

/*!
 * \brief ExceptionHandler Stmt.
 */
class ExceptionHandlerNode : public StmtNode {
 public:
  BaseExpr e;
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("e", &e);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const ExceptionHandlerNode* other, SEqualReducer equal) const {
    return equal(e, other->e) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(e);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "ir.ExceptionHandler";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ExceptionHandlerNode, StmtNode);
};

/*!
 * \brief Managed reference to ExceptionHandlerNode.
 * \sa ExceptionHandlerNode
 */
class ExceptionHandler : public Stmt {
 public:
  MATX_DLL ExceptionHandler(BaseExpr e, Stmt body, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ExceptionHandler, Stmt, ExceptionHandlerNode);
};

/*!
 * \brief TryExcept Stmt.
 */
class TryExceptNode : public StmtNode {
 public:
  Stmt body;
  runtime::Array<ExceptionHandler> handlers;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("body", &body);
    v->Visit("handlers", &handlers);
  }

  bool SEqualReduce(const TryExceptNode* other, SEqualReducer equal) const {
    return equal(body, other->body) && equal(handlers, other->handlers);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(body);
    hash_reduce(handlers);
  }

  static constexpr const char* _type_key = "ir.TryExcept";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TryExceptNode, StmtNode);
};

/*!
 * \brief Managed reference to TryExceptNode.
 * \sa TryExceptNode
 */
class TryExcept : public Stmt {
 public:
  MATX_DLL TryExcept(Stmt body, runtime::Array<ExceptionHandler> handlers, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(TryExcept, Stmt, TryExceptNode);
};

/*!
 * \brief Raise Stmt.
 */
class RaiseNode : public StmtNode {
 public:
  BaseExpr exc;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("exc", &exc);
  }

  bool SEqualReduce(const RaiseNode* other, SEqualReducer equal) const {
    return equal(exc, other->exc);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(exc);
  }

  static constexpr const char* _type_key = "ir.Raise";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(RaiseNode, StmtNode);
};

/*!
 * \brief Managed reference to RaiseNode.
 * \sa RaiseNode
 */
class Raise : public Stmt {
 public:
  MATX_DLL Raise(BaseExpr exc, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Raise, Stmt, RaiseNode);
};

/*!
 * \brief Evaluates an expression.
 *  This is mostly used for putting a Call node into Stmt.
 *
 *  If value do not have side-effect, this node can be safely removed.
 */
class EvaluateNode : public StmtNode {
 public:
  /*! \brief The expression to be evaluated. */
  PrimExpr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("value", &value);
  }

  bool SEqualReduce(const EvaluateNode* other, SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.Evaluate";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(EvaluateNode, StmtNode);
};

/*!
 * \brief Managed reference to EvaluateNode.
 * \sa EvaluateNode
 */
class Evaluate : public Stmt {
 public:
  MATX_DLL explicit Evaluate(PrimExpr value, Span span = Span());

  explicit Evaluate(int value, Span span = Span()) : Evaluate(PrimExpr(value), std::move(span)) {
  }

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Evaluate, Stmt, EvaluateNode);
};

/*! \brief Additional annotation of for loop. */
enum class ForType : int {
  /*! \brief serial execution. */
  Serial = 0,
  /*! \brief parallel execution on CPU. */
  Parallel = 1,
  /*! \brief Vector SIMD loop annotaion. */
  Vectorized = 2,
  /*! \brief Unroll annotation. */
  Unrolled = 3
};

// Kevice api of for loop
// kept for backward compatibility
// consider refactor and remove later.
enum class DeviceAPI : int { None = 0 };

/*!
 * \brief A for loop, with poissible type annotations.
 *
 * \code
 *
 *  for (loop_var = min; loop_var < max; loop_var += step) {
 *    // body
 *  }
 * \endcode
 */
class ForNode : public StmtNode {
 public:
  /*! \brief The loop variable. */
  PrimVar loop_var;
  /*! \brief The temporary loop variable. */
  PrimVar tmp_loop_var;
  /*! \brief The minimum value of iteration. */
  BaseExpr min;
  /*! \brief The maximum value of iteration. */
  BaseExpr max;
  /*! \brief The step value of iteration. */
  BaseExpr step;
  /*! \brief The type of the for loop. */
  ForType for_type;

  /*! \brief The body of the for loop. */
  Stmt body;

  /*! \brief internal use, only for yield. */
  bool yield_mode = false;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("loop_var", &loop_var);
    v->Visit("tmp_loop_var", &tmp_loop_var);
    v->Visit("min", &min);
    v->Visit("step", &step);
    v->Visit("extent", &max);
    v->Visit("for_type", &for_type);
    v->Visit("body", &body);
    v->Visit("yield_mode", &yield_mode);
  }

  bool SEqualReduce(const ForNode* other, SEqualReducer equal) const {
    return equal.DefEqual(loop_var, other->loop_var) &&
           equal.DefEqual(tmp_loop_var, other->tmp_loop_var) && equal(min, other->min) &&
           equal(max, other->max) && equal(step, other->step) && equal(for_type, other->for_type) &&
           equal(body, other->body) && equal(yield_mode, other->yield_mode);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(loop_var);
    hash_reduce.DefHash(tmp_loop_var);
    hash_reduce(min);
    hash_reduce(max);
    hash_reduce(step);
    hash_reduce(for_type);
    hash_reduce(body);
    hash_reduce(yield_mode);
  }

  static constexpr const char* _type_key = "ir.For";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ForNode, StmtNode);
};

/*!
 * \brief Managed reference to ForNode.
 * \sa ForNode
 */
class For : public Stmt {
 public:
  MATX_DLL For(PrimVar loop_var,
               BaseExpr min,
               BaseExpr max,
               BaseExpr step,
               ForType for_type,
               Stmt body,
               Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(For, Stmt, ForNode);
};

class AutoForNode : public StmtNode {
 public:
  /*! \brief The container value of iteration. */
  BaseExpr raw_container;
  /*! \brief The temp container value of iteration. */
  runtime::Array<BaseExpr> eval_containers;
  /*! \brief The loop iter variable. */
  runtime::Array<BaseExpr> iter_vars;  // make_iterable or iter_begin
  /*! \brief The loop iter end variable. */
  runtime::Array<BaseExpr> iter_end_vars;  // has_next or iter_end
  /*! \brief The loop var holder. */
  runtime::Array<BaseExpr> loop_vars_holder;  // for view optimizer
  /*! \brief The loop variable. */
  runtime::Array<BaseExpr> loop_vars;  // x, y, z...
  /*! \brief The body of the for loop. */
  Stmt body;

  /*! \brief internal use, only for yield. */
  bool yield_mode = false;
  runtime::Map<StringRef, BaseExpr> temp_vars;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("iter_var", &iter_vars);
    v->Visit("iter_end_vars", &iter_end_vars);
    v->Visit("loop_vars_holder", &loop_vars_holder);
    v->Visit("loop_var", &loop_vars);
    v->Visit("raw_container", &raw_container);
    v->Visit("container", &eval_containers);
    v->Visit("yield_mode", &yield_mode);
    v->Visit("body", &body);
    v->Visit("temp_vars", &temp_vars);
  }

  bool SEqualReduce(const AutoForNode* other, SEqualReducer equal) const {
    return equal.DefEqual(iter_vars, other->iter_vars) &&
           equal.DefEqual(iter_end_vars, other->iter_end_vars) &&
           equal.DefEqual(loop_vars_holder, other->loop_vars_holder) &&
           equal.DefEqual(loop_vars, other->loop_vars) &&
           equal.DefEqual(temp_vars, other->temp_vars) &&
           equal(raw_container, other->raw_container) &&
           equal(eval_containers, other->eval_containers) && equal(body, other->body) &&
           equal(yield_mode, other->yield_mode);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(iter_vars);
    hash_reduce.DefHash(iter_end_vars);
    hash_reduce.DefHash(loop_vars_holder);
    hash_reduce.DefHash(loop_vars);
    hash_reduce.DefHash(temp_vars);
    hash_reduce(raw_container);
    hash_reduce(eval_containers);
    hash_reduce(yield_mode);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "ir.AutoFor";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AutoForNode, StmtNode);
};

/*!
 * \brief Managed reference to AutoForNode.
 * \sa AutoForNode
 */
class AutoFor : public Stmt {
 public:
  static const char* TEMP_VALUE_VAR_KEY;
  static const char* TEMP_ENUMERATE_POS_VAR_KEY;

  MATX_DLL AutoFor(BaseExpr loop_var, BaseExpr container, Stmt body, Span span = Span())
      : AutoFor(runtime::Array<BaseExpr>{std::move(loop_var)},
                std::move(container),
                std::move(body),
                std::move(span)) {
  }
  MATX_DLL AutoFor(runtime::Array<BaseExpr> loop_vars,
                   BaseExpr container,
                   Stmt body,
                   Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(AutoFor, Stmt, AutoForNode);
};

class WhileNode : public StmtNode {
 public:
  /*! \brief condition. */
  BaseExpr cond;

  /*! \brief The body of the for loop. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("cond", &cond);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const WhileNode* other, SEqualReducer equal) const {
    return equal(cond, other->cond) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(cond);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "ir.While";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(WhileNode, StmtNode);
};

class While : public Stmt {
 public:
  MATX_DLL While(BaseExpr cond, Stmt body, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(While, Stmt, WhileNode);
};

class BreakNode : public StmtNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
  }

  bool SEqualReduce(const BreakNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  static constexpr const char* _type_key = "ir.Break";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(BreakNode, StmtNode);
};

class Break : public Stmt {
 public:
  Break();
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Break, Stmt, BreakNode);
};

class ContinueNode : public StmtNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
  }

  bool SEqualReduce(const ContinueNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  static constexpr const char* _type_key = "ir.Continue";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ContinueNode, StmtNode);
};

class Continue : public Stmt {
 public:
  Continue();
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Continue, Stmt, ContinueNode);
};

/*! \brief HLOYield */
class HLOYieldNode : public StmtNode {
 public:
  /*! \brief the return symbol of the HLOYield, may be a single value or tuple. */
  BaseExpr symbol;
  /*! \brief the String label of the HLOYield, used for codegen goto */
  BaseExpr label;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("symbol", &symbol);
    v->Visit("label", &label);
  }

  bool SEqualReduce(const HLOYieldNode* other, SEqualReducer equal) const {
    // specially handle empty HLOYield as a constant is not a graph node.
    if (symbol.same_as(other->symbol) && label.same_as(other->label)) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(symbol, other->symbol) && equal(label, other->label);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(symbol);
    hash_reduce(label);
  }

  static constexpr const char* _type_key = "ir.HLOYield";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(HLOYieldNode, StmtNode);
};

class HLOYield : public Stmt {
 public:
  /*!
   * \brief The constructor
   * \param symbol The symbol fields of a HLOYield.
   * \param label The label fields of a HLOYield.
   * \param span The source span of the expression.
   */
  MATX_DLL explicit HLOYield(BaseExpr symbol, BaseExpr label, Span span = Span());
  MATX_DLL explicit HLOYield(BaseExpr symbol, Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(HLOYield, Stmt, HLOYieldNode);
};

// overload printing of for type.
MATX_DLL std::ostream& operator<<(std::ostream& os, ForType for_type);

}  // namespace ir
}  // namespace matxscript
