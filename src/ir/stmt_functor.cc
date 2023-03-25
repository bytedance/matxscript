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
 * \file stmt_functor.cc
 */
#include <matxscript/ir/stmt_functor.h>
#include "functor_common.h"

#include <functional>
#include <unordered_set>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::Downcast;
using ::matxscript::runtime::GetRef;
using ::matxscript::runtime::NativeFunction;

void StmtVisitor::VisitStmt_(const AllocaVarStmtNode* op) {
  this->VisitExpr(op->var);
  this->VisitExpr(op->init_value);
}

void StmtVisitor::VisitStmt_(const AssignStmtNode* op) {
  this->VisitExpr(op->lhs);
  this->VisitExpr(op->rhs);
}

void StmtVisitor::VisitStmt_(const ReturnStmtNode* op) {
  this->VisitExpr(op->value);
}

void StmtVisitor::VisitStmt_(const ForNode* op) {
  this->VisitExpr(op->loop_var);
  this->VisitExpr(op->tmp_loop_var);
  this->VisitExpr(op->min);
  this->VisitExpr(op->max);
  this->VisitExpr(op->step);
  this->VisitStmt(op->body);
}
void StmtVisitor::VisitStmt_(const AutoForNode* op) {
  for (auto& loop_var : op->loop_vars) {
    this->VisitExpr(loop_var);
  }
  for (auto& loop_var_holder : op->loop_vars_holder) {
    this->VisitExpr(loop_var_holder);
  }
  for (auto& temp_var : op->temp_vars) {
    this->VisitExpr(temp_var.second);
  }
  for (auto& iter_var : op->iter_vars) {
    this->VisitExpr(iter_var);
  }
  for (auto& iter_end_var : op->iter_end_vars) {
    this->VisitExpr(iter_end_var);
  }
  for (auto& cons_var : op->eval_containers) {
    this->VisitExpr(cons_var);
  }
  this->VisitExpr(op->raw_container);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const WhileNode* op) {
  this->VisitExpr(op->cond);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const BreakNode* op) {
}

void StmtVisitor::VisitStmt_(const ContinueNode* op) {
}

void StmtVisitor::VisitStmt_(const IfThenElseNode* op) {
  this->VisitExpr(op->condition);
  this->VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    this->VisitStmt(op->else_case);
  }
}

void StmtVisitor::VisitStmt_(const ExceptionHandlerNode* op) {
  if (op->e.defined()) {
    this->VisitExpr(op->e);
  }
  if (op->body.defined()) {
    this->VisitStmt(op->body);
  }
}

void StmtVisitor::VisitStmt_(const TryExceptNode* op) {
  if (op->body.defined()) {
    this->VisitStmt(op->body);
  }
  if (!op->handlers.empty()) {
    for (auto& handler : op->handlers) {
      this->VisitStmt(handler);
    }
  }
}

void StmtVisitor::VisitStmt_(const RaiseNode* op) {
  if (op->exc.defined()) {
    this->VisitExpr(op->exc);
  }
}

void StmtVisitor::VisitStmt_(const AssertStmtNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->message);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const SeqStmtNode* op) {
  VisitArray(op->seq, [this](const Stmt& s) { this->VisitStmt(s); });
}

void StmtVisitor::VisitStmt_(const ExprStmtNode* op) {
  this->VisitExpr(op->expr);
}

void StmtVisitor::VisitStmt_(const HLOYieldNode* op) {
  this->VisitExpr(op->symbol);
  this->VisitExpr(op->label);
}

void StmtVisitor::VisitStmt_(const ComputeBlockNode* op) {
  auto fn_visit_buffer_region = [this](const BufferRegion& s) {
    for (const auto& range : s->region) {
      this->VisitExpr(range);
    }
  };
  VisitArray(op->iter_vars, [this](const PrimIterVar& iter_var) {
    this->VisitExpr(iter_var->dom);
    this->VisitExpr(iter_var->var);
  });
  VisitArray(op->reads, fn_visit_buffer_region);
  VisitArray(op->writes, fn_visit_buffer_region);
  VisitArray(op->match_buffers,
             [fn_visit_buffer_region](const MatchBufferRegion& match_buffer_region) {
               fn_visit_buffer_region(match_buffer_region->source);
             });
  if (op->init.defined()) {
    this->VisitStmt(op->init.value());
  }
  this->VisitStmt(op->body);
}

class StmtMutator::Internal {
 public:
  /*!
   * \brief Mutate array's element by fmutate function.
   *
   * \note Use extra care for copy on write setting.
   *
   * In particular, consider the following case of two reference chains:
   * - strongref0 -> loop0 -> loop1 -> loop2
   * - strongref1 -> loop3 -> loop1 -> loop2
   *
   * Think of the case of calling MutateArray on loop1->loop2(as const reference).
   * When both strongref0 and strongref1 exists, the context does not allow copy
   * on write, even though loop1 uniquely refers to loop2.
   *
   * \param self The pointer to the mutator.
   * \param arr Array to be mutated, const reference is used to allow copy on write
   *            mutation in a recursive visitor.
   * \param fmutate The mutator function.
   * \return The mutated array, a new copy can be created.
   */
  template <typename T, typename F>
  static Array<T> MutateArray(StmtMutator* self, const Array<T>& arr, F fmutate) {
    if (self->allow_copy_on_write_ && arr.unique()) {
      // if we allow copy on write, we can directly
      // call the inplace mutate function.
      const_cast<Array<T>&>(arr).MutateByApply(fmutate);
      return arr;
    } else {
      bool allow_cow = false;
      std::swap(allow_cow, self->allow_copy_on_write_);
      Array<T> copy = arr.Map(fmutate);
      std::swap(allow_cow, self->allow_copy_on_write_);
      return copy;
    }
  }

  static Array<PrimExpr> Mutate(StmtMutator* self, const Array<PrimExpr>& arr) {
    auto fmutate = [self](const PrimExpr& e) { return self->VisitExpr(e); };
    return MutateArray(self, arr, fmutate);
  }

  static Array<HLOExpr> Mutate(StmtMutator* self, const Array<HLOExpr>& arr) {
    auto fmutate = [self](const HLOExpr& e) { return self->VisitExpr(e); };
    return MutateArray(self, arr, fmutate);
  }

  static Array<Stmt> Mutate(StmtMutator* self, const Array<Stmt>& arr) {
    auto fmutate = [self](const Stmt& s) { return self->VisitStmt(s); };
    return MutateArray(self, arr, fmutate);
  }

  static Array<PrimIterVar> Mutate(StmtMutator* self, const Array<PrimIterVar>& arr) {
    auto fn_mutate = [self](const PrimIterVar& iter_var) {
      RangeExpr dom = Downcast<RangeExpr>(self->VisitExpr(iter_var->dom));
      // TODO: Why is tvm ignoring visit iter_var->var？
      PrimVar var = Downcast<PrimVar>(self->VisitExpr(iter_var->var));
      if (dom.same_as(iter_var->dom) && var.same_as(iter_var->var)) {
        return iter_var;
      } else {
        return PrimIterVar(std::move(dom), std::move(var));
      }
    };
    return MutateArray(self, arr, fn_mutate);
  }

  static Array<BufferRegion> Mutate(StmtMutator* self, const Array<BufferRegion>& arr) {
    auto fmutate = [self](const BufferRegion& buffer_region) {
      Array<RangeExpr> region = Downcast<Array<RangeExpr>>(
          Mutate(self, runtime::Downcast<Array<HLOExpr>>(buffer_region->region)));
      if (region.same_as(buffer_region->region)) {
        return buffer_region;
      } else {
        return BufferRegion(buffer_region->buffer, region);
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static Array<MatchBufferRegion> Mutate(StmtMutator* self, const Array<MatchBufferRegion>& arr) {
    auto fmutate = [self](const MatchBufferRegion& match_buffer_region) {
      Array<RangeExpr> region = Downcast<Array<RangeExpr>>(
          Mutate(self, runtime::Downcast<Array<HLOExpr>>(match_buffer_region->source->region)));
      if (region.same_as(match_buffer_region->source->region)) {
        return match_buffer_region;
      } else {
        return MatchBufferRegion(match_buffer_region->buffer,
                                 BufferRegion(match_buffer_region->source->buffer, region));
      }
    };
    return MutateArray(self, arr, fmutate);
  }
};

Stmt StmtMutator::VisitStmt_(const AllocaVarStmtNode* op) {
  BaseExpr var = this->VisitExpr(op->var);
  BaseExpr init_value = this->VisitExpr(op->init_value);
  if (var.same_as(op->var) && init_value.same_as(op->init_value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->var = std::move(var);
    n->init_value = std::move(init_value);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AssignStmtNode* op) {
  BaseExpr lhs = this->VisitExpr(op->lhs);
  BaseExpr rhs = this->VisitExpr(op->rhs);
  if (lhs.same_as(op->lhs) && rhs.same_as(op->rhs)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->lhs = std::move(lhs);
    n->rhs = std::move(rhs);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ReturnStmtNode* op) {
  BaseExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ForNode* op) {
  BaseExpr loop_var = this->VisitExpr(op->loop_var);
  BaseExpr tmp_loop_var = this->VisitExpr(op->tmp_loop_var);
  BaseExpr min = this->VisitExpr(op->min);
  BaseExpr max = this->VisitExpr(op->max);
  BaseExpr step = this->VisitExpr(op->step);
  Stmt body = this->VisitStmt(op->body);
  if (loop_var.same_as(op->loop_var) && tmp_loop_var.same_as(op->tmp_loop_var) &&
      min.same_as(op->min) && max.same_as(op->max) && step.same_as(op->step) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->loop_var = Downcast<PrimVar>(loop_var);
    n->tmp_loop_var = Downcast<PrimVar>(tmp_loop_var);
    n->min = std::move(min);
    n->max = std::move(max);
    n->step = std::move(step);
    n->body = std::move(body);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AutoForNode* op) {
  bool same = true;

  BaseExpr raw_container = this->VisitExpr(op->raw_container);
  same &= raw_container.same_as(op->raw_container);

  Stmt body = this->VisitStmt(op->body);
  same &= body.same_as(op->body);

  Array<BaseExpr> loop_vars;
  for (auto i = 0; i < op->loop_vars.size(); ++i) {
    BaseExpr loop_var = this->VisitExpr(op->loop_vars[i]);
    same &= loop_var.same_as(op->loop_vars[i]);
    loop_vars.push_back(std::move(loop_var));
  }

  Array<BaseExpr> loop_vars_holder;
  for (auto i = 0; i < op->loop_vars_holder.size(); ++i) {
    BaseExpr loop_var_holder = this->VisitExpr(op->loop_vars_holder[i]);
    same &= loop_var_holder.same_as(op->loop_vars_holder[i]);
    loop_vars_holder.push_back(std::move(loop_var_holder));
  }

  Array<BaseExpr> iter_vars;
  for (auto i = 0; i < op->iter_vars.size(); ++i) {
    BaseExpr iter_var = this->VisitExpr(op->iter_vars[i]);
    same &= iter_var.same_as(op->iter_vars[i]);
    iter_vars.push_back(std::move(iter_var));
  }

  Array<BaseExpr> iter_end_vars;
  for (auto i = 0; i < op->iter_end_vars.size(); ++i) {
    BaseExpr iter_end_var = this->VisitExpr(op->iter_end_vars[i]);
    same &= iter_end_var.same_as(op->iter_end_vars[i]);
    iter_end_vars.push_back(std::move(iter_end_var));
  }

  Array<BaseExpr> eval_containers;
  for (auto i = 0; i < op->eval_containers.size(); ++i) {
    BaseExpr eval_cons = this->VisitExpr(op->eval_containers[i]);
    same &= eval_cons.same_as(op->eval_containers[i]);
    eval_containers.push_back(std::move(eval_cons));
  }

  Map<StringRef, BaseExpr> temp_vars;
  for (auto temp_var_iter : op->temp_vars) {
    BaseExpr new_temp_var = this->VisitExpr(temp_var_iter.second);
    same &= new_temp_var.same_as(temp_var_iter.second);
    temp_vars.Set(temp_var_iter.first, new_temp_var);
  }

  if (same) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->yield_mode = op->yield_mode;
    n->loop_vars = std::move(loop_vars);
    n->loop_vars_holder = std::move(loop_vars_holder);
    n->temp_vars = std::move(temp_vars);
    n->iter_vars = std::move(iter_vars);
    n->iter_end_vars = std::move(iter_end_vars);
    n->eval_containers = std::move(eval_containers);
    n->raw_container = std::move(raw_container);
    n->body = std::move(body);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const WhileNode* op) {
  BaseExpr cond = this->VisitExpr(op->cond);
  Stmt body = this->VisitStmt(op->body);
  if (cond.same_as(op->cond) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->cond = cond;
    n->body = body;
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BreakNode* op) {
  return GetRef<Stmt>(op);
}

Stmt StmtMutator::VisitStmt_(const ContinueNode* op) {
  return GetRef<Stmt>(op);
}

Stmt StmtMutator::VisitStmt_(const IfThenElseNode* op) {
  BaseExpr condition = this->VisitExpr(op->condition);
  Stmt then_case = this->VisitStmt(op->then_case);
  Stmt else_case;
  if (op->else_case.defined()) {
    else_case = this->VisitStmt(op->else_case);
  }
  if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ExceptionHandlerNode* op) {
  BaseExpr e;
  Stmt body;
  if (op->e.defined()) {
    e = this->VisitExpr(op->e);
  }
  if (op->body.defined()) {
    body = this->VisitStmt(op->body);
  }
  if (e.same_as(op->e) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->e = std::move(e);
    n->body = std::move(body);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const TryExceptNode* op) {
  Stmt body;
  if (op->body.defined()) {
    body = this->VisitStmt(op->body);
  }
  auto fmutate = [this](const ExceptionHandler& s) {
    return Downcast<ExceptionHandler>(this->VisitStmt(s));
  };
  Array<ExceptionHandler> handlers = Internal::MutateArray(this, op->handlers, fmutate);
  if (body.same_as(op->body) && handlers.same_as(op->handlers)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->body = std::move(body);
    n->handlers = std::move(handlers);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const RaiseNode* op) {
  BaseExpr exc;
  if (op->exc.defined()) {
    exc = this->VisitExpr(op->exc);
  }
  if (exc.same_as(op->exc)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->exc = std::move(exc);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SeqStmtNode* op) {
  Array<Stmt> seq = Internal::Mutate(this, op->seq);
  if (seq.same_as(op->seq)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->seq = std::move(seq);
    n->span = op->span;
    return Stmt(n);
  }
}

// advanced visit function for seqstmt.
Stmt StmtMutator::VisitSeqStmt_(const SeqStmtNode* op,
                                bool flatten_before_visit,
                                std::function<Stmt(const Stmt&)> fmutate) {
  if (flatten_before_visit) {
    // Pass 1, check if we need to flatten.
    bool need_flatten = false;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      Stmt tmp = (*op)[i];
      if (tmp.as<SeqStmtNode>())
        need_flatten = true;
    }
    flatten_before_visit = need_flatten;
  }
  // function to run the visit.
  auto frunvisit = [&](const SeqStmtNode* op) {
    Array<Stmt> seq = fmutate != nullptr ? Internal::MutateArray(this, op->seq, fmutate)
                                         : Internal::Mutate(this, op->seq);
    if (seq.same_as(op->seq)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->span = op->span;
      n->seq = std::move(seq);
      return Stmt(n);
    }
  };
  if (flatten_before_visit) {
    Array<Stmt> seq;
    SeqStmt::Flattener flattener(&seq);
    flattener(0, op->seq);
    // NOTE: If copy on write is allowed
    // the assignment to seq below will
    // destruct the original seq.
    //
    // Such destruction removes duplicated reference
    // count to children and still enables COW for
    // child Stmt.
    ObjectPtr<SeqStmtNode> n = CopyOnWrite(op);
    n->seq = std::move(seq);
    n->span = op->span;
    return frunvisit(n.operator->());
  } else {
    return frunvisit(op);
  }
}

Stmt StmtMutator::VisitStmt_(const AssertStmtNode* op) {
  BaseExpr condition = this->VisitExpr(op->condition);
  BaseExpr message = this->VisitExpr(op->message);
  Stmt body = this->VisitStmt(op->body);

  if (condition.same_as(op->condition) && message.same_as(op->message) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->message = std::move(message);
    n->body = std::move(body);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ExprStmtNode* op) {
  BaseExpr value = this->VisitExpr(op->expr);
  if (value.same_as(op->expr)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->expr = std::move(value);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const HLOYieldNode* op) {
  BaseExpr symbol = this->VisitExpr(op->symbol);
  BaseExpr label = this->VisitExpr(op->label);
  if (symbol.same_as(op->symbol) && label.same_as(op->label)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->symbol = std::move(symbol);
    n->label = std::move(label);
    n->span = op->span;
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ComputeBlockNode* op) {
  Array<PrimIterVar> iter_vars = Internal::Mutate(this, op->iter_vars);
  Array<BufferRegion> reads = Internal::Mutate(this, op->reads);
  Array<BufferRegion> writes = Internal::Mutate(this, op->writes);
  Array<MatchBufferRegion> match_buffers = Internal::Mutate(this, op->match_buffers);
  Optional<Stmt> init = NullOpt;
  if (op->init.defined()) {
    init = VisitStmt(op->init.value());
  }
  Stmt body = VisitStmt(op->body);
  if (iter_vars.same_as(op->iter_vars) && reads.same_as(op->reads) && writes.same_as(op->writes) &&
      body.same_as(op->body) && init.same_as(op->init) &&
      match_buffers.same_as(op->match_buffers)) {
    return GetRef<ComputeBlock>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->iter_vars = std::move(iter_vars);
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->body = std::move(body);
    n->init = std::move(init);
    n->match_buffers = std::move(match_buffers);
    return Stmt(n);
  }
}

void StmtVisitor::VisitStmt_(const PrimFuncNode* op) {
  // TODO: fix visitor
  // this->VisitSpan(op->span);
  // this->VisitType(op->ret_type);
  for (auto param : op->params) {
    this->VisitExpr(param);
  }
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const FunctionNode* op) {
  // this->VisitSpan(op->span);
  // this->VisitType(op->ret_type);
  for (auto param : op->params) {
    this->VisitExpr(param);
  }
  this->VisitStmt(op->body);
  if (op->type_params.defined()) {
    for (auto param_t : op->type_params) {
      if (param_t.defined()) {
        // this->VisitType(param_t);
      }
    }
  }
}

void StmtExprVisitor::VisitExpr_(const LambdaFunctionNode* op) {
  this->VisitSpan(op->span);
  this->VisitType(op->ret_type);
  for (auto param : op->params) {
    this->VisitExpr(param);
  }
  this->VisitStmt(op->body);
  if (op->captures.defined()) {
    for (auto cap : op->captures) {
      if (cap.defined()) {
        this->VisitExpr(cap);
      }
    }
  }
}

Stmt StmtMutator::VisitStmt_(const PrimFuncNode* op) {
  bool all_fields_unchanged = true;
  // ret_type
  Type ret_type = op->ret_type;  // TODO: this->VisitType(op->ret_type);
  all_fields_unchanged &= ret_type.same_as(op->ret_type);

  // params
  Array<PrimVar> params;
  for (auto param : op->params) {
    auto new_param = this->VisitExpr(param);
    params.push_back(Downcast<PrimVar>(new_param));
    all_fields_unchanged &= new_param.same_as(param);
  }

  Array<PrimExpr> default_params;
  for (auto param : op->default_params) {
    auto new_param = this->VisitExpr(param);
    default_params.push_back(Downcast<PrimExpr>(new_param));
    all_fields_unchanged &= new_param.same_as(param);
  }

  // body
  Stmt body = this->VisitStmt(op->body);
  all_fields_unchanged &= body.same_as(op->body);

  // attrs can not be mutable
  if (all_fields_unchanged) {
    return GetRef<PrimFunc>(op);
  } else {
    return PrimFunc(params, default_params, body, ret_type, op->attrs, op->span);
  }
}

Stmt StmtMutator::VisitStmt_(const FunctionNode* op) {
  bool all_fields_unchanged = true;
  // ret_type
  Type ret_type = op->ret_type;  // TODO: this->VisitType(op->ret_type);
  all_fields_unchanged &= ret_type.same_as(op->ret_type);

  // params
  Array<BaseExpr> params;
  for (auto param : op->params) {
    auto new_param = this->VisitExpr(param);
    params.push_back(new_param);
    all_fields_unchanged &= new_param.same_as(param);
  }

  Array<BaseExpr> default_params;
  for (auto param : op->default_params) {
    auto new_param = this->VisitExpr(param);
    default_params.push_back(new_param);
    all_fields_unchanged &= new_param.same_as(param);
  }

  // body
  Stmt body = this->VisitStmt(op->body);
  all_fields_unchanged &= body.same_as(op->body);

  // type_params
  Array<TypeVar> type_params;
  for (auto param_t : op->type_params) {
    if (param_t.defined()) {
      auto new_param_t = param_t;  // TODO: this->VisitType(param_t);
      type_params.push_back(Downcast<TypeVar>(new_param_t));
      all_fields_unchanged &= new_param_t.same_as(param_t);
    } else {
      type_params.push_back(param_t);
    }
  }

  // attrs can not be mutable
  if (all_fields_unchanged) {
    return GetRef<Function>(op);
  } else {
    return Function(params, default_params, body, ret_type, type_params, op->attrs, op->span);
  }
}

HLOExpr StmtExprMutator::VisitExpr_(const LambdaFunctionNode* op) {
  bool all_fields_unchanged = true;
  // ret_type
  Type ret_type = this->VisitType(op->ret_type);
  all_fields_unchanged &= ret_type.same_as(op->ret_type);

  // params
  Array<BaseExpr> params;
  for (auto param : op->params) {
    auto new_param = this->VisitExpr(param);
    params.push_back(new_param);
    all_fields_unchanged &= new_param.same_as(param);
  }

  Array<BaseExpr> captures;
  for (auto param : op->captures) {
    auto new_param = this->VisitExpr(param);
    captures.push_back(new_param);
    all_fields_unchanged &= new_param.same_as(param);
  }

  // body
  Stmt body = this->VisitStmt(op->body);
  all_fields_unchanged &= body.same_as(op->body);

  // attrs can not be mutable
  if (all_fields_unchanged) {
    return GetRef<LambdaFunction>(op);
  } else {
    return LambdaFunction(captures, params, body, ret_type, op->span);
  }
}

// Implementations of IRTransform, PostOrderVisit and Substitute
class IRApplyVisit : public StmtExprVisitor {
 public:
  explicit IRApplyVisit(std::function<void(const ObjectRef&)> f) : f_(f) {
  }

  void VisitExpr(const PrimExpr& node) final {
    if (visited_.count(node.get()) != 0)
      return;
    visited_.insert(node.get());
    ExprVisitor::VisitExpr(node);
    f_(node);
  }

  void VisitStmt(const Stmt& node) final {
    if (visited_.count(node.get()) != 0)
      return;
    visited_.insert(node.get());
    StmtVisitor::VisitStmt(node);
    f_(node);
  }

 private:
  std::function<void(const ObjectRef&)> f_;
  std::unordered_set<const Object*> visited_;
};

void PostOrderVisit(const ObjectRef& node, std::function<void(const ObjectRef&)> fvisit) {
  if (node.as<StmtNode>()) {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<Stmt>(node));
  } else {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<PrimExpr>(node));
  }
}

class IRTransformer final : public StmtExprMutator {
 public:
  IRTransformer(const runtime::NativeFunction& f_preorder,
                const runtime::NativeFunction& f_postorder,
                const std::unordered_set<uint32_t>& only_enable)
      : f_preorder_(f_preorder), f_postorder_(f_postorder), only_enable_(only_enable) {
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    return MutateInternal<Stmt>(stmt, [this](const Stmt& s) { return this->BaseVisitStmt(s); });
  }
  PrimExpr VisitExpr(const PrimExpr& expr) final {
    return MutateInternal<PrimExpr>(expr,
                                    [this](const PrimExpr& e) { return this->BaseVisitExpr(e); });
  }

 private:
  // NOTE: redirect to parent's call
  // This is used to get around limitation of gcc-4.8
  Stmt BaseVisitStmt(const Stmt& s) {
    return StmtMutator::VisitStmt(s);
  }
  PrimExpr BaseVisitExpr(const PrimExpr& e) {
    return ExprMutator::VisitExpr(e);
  }

  template <typename T, typename F>
  T MutateInternal(const T& node, F fmutate) {
    if (only_enable_.size() && !only_enable_.count(node->type_index())) {
      return fmutate(node);
    }
    if (f_preorder_ != nullptr) {
      T pre = f_preorder_({node}).template As<T>();
      if (pre.defined())
        return pre;
    }
    T new_node = fmutate(node);
    if (f_postorder_ != nullptr) {
      T post = f_postorder_({new_node}).template As<T>();
      if (post.defined())
        return post;
    }
    return new_node;
  }
  // The functions
  const runtime::NativeFunction& f_preorder_;
  const runtime::NativeFunction& f_postorder_;
  // type indices enabled.
  const std::unordered_set<uint32_t>& only_enable_;
};

Stmt IRTransform(Stmt ir_node,
                 const runtime::NativeFunction& f_preorder,
                 const runtime::NativeFunction& f_postorder,
                 Optional<Array<StringRef>> only_enable) {
  std::unordered_set<uint32_t> only_type_index;
  if (only_enable.defined()) {
    for (auto s : only_enable.value()) {
      only_type_index.insert(Object::TypeKey2Index(s.c_str()));
    }
  }
  IRTransformer transform(f_preorder, f_postorder, only_type_index);
  return transform(std::move(ir_node));
}

class IRSubstitue : public StmtExprMutator {
 public:
  explicit IRSubstitue(std::function<Optional<BaseExpr>(const BaseExpr&)> vmap)
      : vmap_base_(std::move(vmap)) {
  }
  explicit IRSubstitue(std::function<Optional<PrimExpr>(const PrimVar&)> vmap) {
    vmap_base_ = [&vmap](const BaseExpr& var) -> Optional<BaseExpr> {
      if (var->IsInstance<PrimVarNode>()) {
        PrimVar prim_var = Downcast<PrimVar>(var);
        auto ret = vmap(prim_var);
        if (ret.defined())
          return ret.value();
        return std::move(var);
      }
      return var;
    };
  }

  explicit IRSubstitue(std::function<Optional<HLOExpr>(const HLOVar&)> vmap) {
    vmap_base_ = [&vmap](const BaseExpr& var) -> Optional<BaseExpr> {
      if (var->IsInstance<HLOVarNode>()) {
        HLOVar hlo_var = Downcast<HLOVar>(var);
        auto ret = vmap(hlo_var);
        if (ret.defined())
          return ret.value();
        return std::move(var);
      }
      return var;
    };
  }

  PrimExpr VisitExpr_(const PrimVarNode* op) final {
    PrimVar var = GetRef<PrimVar>(op);
    auto ret = vmap_base_(var);
    if (ret.defined())
      return Downcast<PrimExpr>(ret.value());
    return std::move(var);
  }

  HLOExpr VisitExpr_(const HLOVarNode* op) final {
    HLOVar var = GetRef<HLOVar>(op);
    auto ret = vmap_base_(var);
    if (ret.defined())
      return Downcast<HLOExpr>(ret.value());
    return std::move(var);
  }

 private:
  std::function<Optional<BaseExpr>(const BaseExpr&)> vmap_base_;
};

Stmt Substitute(Stmt stmt, std::function<Optional<PrimExpr>(const PrimVar&)> vmap) {
  return IRSubstitue(vmap)(std::move(stmt));
}

Stmt Substitute(Stmt stmt, std::function<Optional<HLOExpr>(const HLOVar&)> vmap) {
  return IRSubstitue(vmap)(std::move(stmt));
}

Stmt Substitute(Stmt stmt, std::function<Optional<BaseExpr>(const BaseExpr&)> vmap) {
  return IRSubstitue(vmap)(std::move(stmt));
}

PrimExpr Substitute(PrimExpr expr, std::function<Optional<PrimExpr>(const PrimVar&)> vmap) {
  return IRSubstitue(vmap)(std::move(expr));
}

HLOExpr Substitute(HLOExpr expr, std::function<Optional<HLOExpr>(const HLOVar&)> vmap) {
  return IRSubstitue(vmap)(std::move(expr));
}

BaseExpr Substitute(BaseExpr expr, std::function<Optional<BaseExpr>(const BaseExpr&)> vmap) {
  return IRSubstitue(vmap)(std::move(expr));
}

}  // namespace ir
}  // namespace matxscript
