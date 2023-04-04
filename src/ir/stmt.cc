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
 * \file matx/ir/stmt.cc
 */
#include <matxscript/ir/stmt.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/with.h>
#include <matxscript/ir/hlo_expr.h>
#include <matxscript/ir/hlo_var.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/ir/prim_var.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/ir/printer/utils.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

namespace printer {
bool AllowConciseScoping(const IRDocsifier& d) {
  MXCHECK(!d->frames.empty());
  if (const auto* f = d->frames.back().as<IRFrameNode>()) {
    return f->allow_concise_scoping;
  }
  MXLOG(FATAL) << "NotImplementedError: fragment printing";
  return false;
}
}  // namespace printer

using ::matxscript::runtime::Downcast;
using ::matxscript::runtime::make_object;
using ::matxscript::runtime::String;
using namespace ::matxscript::ir::printer;

// ExprStmt
ExprStmt::ExprStmt(BaseExpr expr, Span span) {
  ObjectPtr<ExprStmtNode> node = make_object<ExprStmtNode>();
  node->expr = std::move(expr);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.ExprStmt").set_body_typed([](BaseExpr expr, Span span = Span()) {
  return ExprStmt(std::move(expr), span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(ExprStmtNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ExprStmt>("", [](ExprStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      auto expr = d->AsDoc<Doc>(stmt->expr, p->Attr("expr"));
      if (expr->IsInstance<StmtDocNode>()) {
        // a[x] = kk
        return expr;
      }
      return ExprStmtDoc(Downcast<ExprDoc>(expr));
    });

// AllocaVarStmt
AllocaVarStmt::AllocaVarStmt(StringRef name, Type ty, BaseExpr init_value, Span span) {
  ObjectPtr<AllocaVarStmtNode> node = make_object<AllocaVarStmtNode>();
  if (IsRuntimeDataType(ty)) {
    node->var = std::move(PrimVar(name, ty, span));
  } else {
    node->var = std::move(HLOVar(name, ty, span));
  }
  node->init_value = std::move(init_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.AllocaVarStmt")
    .set_body_typed([](StringRef name, Type ty, BaseExpr init_value, Span span = Span()) {
      return AllocaVarStmt(name, ty, init_value, span);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir._GetVarFromAllocaVarStmt").set_body_typed([](AllocaVarStmt stmt) {
  return stmt->var;
});

MATXSCRIPT_REGISTER_NODE_TYPE(AllocaVarStmtNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<AllocaVarStmt>("", [](AllocaVarStmt s, ObjectPath p, IRDocsifier d) -> Doc {
      ObjectPath var_p = p->Attr("var");
      auto lhs = d->AsDoc<ExprDoc>(s->var, var_p);
      Optional<ExprDoc> rhs;
      Optional<ExprDoc> annotation;
      if (s->init_value.defined()) {
        rhs = d->AsDoc<ExprDoc>(s->init_value, p->Attr("init_value"));
      }
      annotation = d->AsDoc<ExprDoc>(s->var->checked_type_, var_p->Attr("checked_type_"));
      return AssignDoc(lhs, rhs, annotation);
    });

// AssignStmt
AssignStmt::AssignStmt(BaseExpr lhs, BaseExpr rhs, Span span) {
  MXCHECK(lhs.defined());
  MXCHECK(rhs.defined());

  ObjectPtr<AssignStmtNode> node = make_object<AssignStmtNode>();
  node->lhs = std::move(lhs);
  node->rhs = std::move(rhs);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.AssignStmt")
    .set_body_typed([](BaseExpr lhs, BaseExpr rhs, Span span = Span()) {
      return AssignStmt(lhs, rhs, span);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(AssignStmtNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<AssignStmt>("", [](AssignStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      auto lhs = d->AsDoc<ExprDoc>(stmt->lhs, p->Attr("lhs"));
      Optional<ExprDoc> rhs = d->AsDoc<ExprDoc>(stmt->rhs, p->Attr("rhs"));
      return AssignDoc(lhs, rhs, NullOpt);
    });

// // Return
ReturnStmt::ReturnStmt(BaseExpr value, Span span) {
  ObjectPtr<ReturnStmtNode> node = make_object<ReturnStmtNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.ReturnStmt").set_body_typed([](BaseExpr value, Span span = Span()) {
  return ReturnStmt(value, span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(ReturnStmtNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ReturnStmt>("", [](ReturnStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      auto value = d->AsDoc<ExprDoc>(stmt->value, p->Attr("value"));
      return ReturnDoc(value);
    });

// AssertStmt
AssertStmt::AssertStmt(BaseExpr condition, BaseExpr message, Stmt body, Span span) {
  MXCHECK(condition.defined());
  MXCHECK(message->checked_type() == PrimType(runtime::DataType::Int(32)) ||
          message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:" << message << "\n";

  ObjectPtr<AssertStmtNode> node = make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(AssertStmtNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.AssertStmt")
    .set_body_typed([](BaseExpr condition, ObjectRef message, Stmt body, Span span = Span()) {
      return AssertStmt(condition, Downcast<BaseExpr>(message), body, span);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::AssertStmt>("", [](ir::AssertStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      bool concise = AllowConciseScoping(d);
      ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
      ExprDoc msg = d->AsDoc<ExprDoc>(stmt->message, p->Attr("message"));
      With<IRFrame> f(d, stmt);
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
      if (concise) {
        Array<StmtDoc>* stmts = &(*f)->stmts;
        stmts->insert(stmts->begin(), AssertDoc(cond, msg));
        return StmtBlockDoc(*stmts);
      }
      return ScopeDoc(NullOpt, Dialect(d, "Assert")->Call({cond, msg}), (*f)->stmts);
    });

// For
For::For(PrimVar loop_var,
         BaseExpr min,
         BaseExpr max,
         BaseExpr step,
         ForType for_type,
         Stmt body,
         Span span) {
  MXCHECK(min.defined());
  MXCHECK(max.defined());
  MXCHECK(step.defined());
  MXCHECK(loop_var.dtype().is_scalar());
  MXCHECK(body.defined());
  if (auto* min_node = min.as<PrimExprNode>()) {
    MXCHECK(min_node->dtype.is_scalar());
  } else {
    auto hlo_min = Downcast<HLOExpr>(min);
    MXCHECK(hlo_min->checked_type().defined());
    auto min_type_node = hlo_min->checked_type().as<PrimTypeNode>();
    MXCHECK(min_type_node != nullptr && min_type_node->dtype.is_scalar())
        << "[ir.For] min is not Prim scalar";
  }
  if (auto* max_node = max.as<PrimExprNode>()) {
    MXCHECK(max_node->dtype.is_scalar());
  } else {
    auto hlo_max = Downcast<HLOExpr>(max);
    MXCHECK(hlo_max->checked_type().defined());
    auto max_type_node = hlo_max->checked_type().as<PrimTypeNode>();
    MXCHECK(max_type_node != nullptr && max_type_node->dtype.is_scalar())
        << "[ir.For] max is not Prim scalar";
  }
  if (auto* step_node = step.as<PrimExprNode>()) {
    MXCHECK(step_node->dtype.is_scalar());
  } else {
    auto hlo_step = Downcast<HLOExpr>(step);
    MXCHECK(hlo_step->checked_type().defined());
    auto step_type_node = hlo_step->checked_type().as<PrimTypeNode>();
    MXCHECK(step_type_node != nullptr && step_type_node->dtype.is_scalar())
        << "[ir.For] step is not Prim scalar";
  }

  ObjectPtr<ForNode> node = make_object<ForNode>();
  node->tmp_loop_var = PrimVar(loop_var->name_hint + "_iter_", loop_var.dtype());
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->max = std::move(max);
  node->step = std::move(step);
  node->for_type = for_type;
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.For").set_body_typed([](PrimVar loop_var,
                                                       BaseExpr min,
                                                       BaseExpr max,
                                                       BaseExpr step,
                                                       int for_type,
                                                       Stmt body,
                                                       Span span = Span()) {
  return For(loop_var, min, max, step, static_cast<ForType>(for_type), body, span);
});

MATXSCRIPT_REGISTER_NODE_TYPE(ForNode);

std::ostream& operator<<(std::ostream& out, ForType type) {  // NOLINT(*)
  switch (type) {
    case ForType::Serial:
      out << "for";
      break;
    case ForType::Parallel:
      out << "parallel";
      break;
    case ForType::Unrolled:
      out << "unrolled";
      break;
    case ForType::Vectorized:
      out << "vectorized";
      break;
  }
  return out;
}

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::For>("", [](ir::For loop, ObjectPath loop_p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, loop);
      ExprDoc lhs = DefineVar(loop->loop_var, *f, d);
      Optional<ExprDoc> min = NullOpt;
      Optional<ExprDoc> max = NullOpt;
      Optional<ExprDoc> step = NullOpt;
      min = d->AsDoc<ExprDoc>(loop->min, loop_p->Attr("min"));
      max = d->AsDoc<ExprDoc>(loop->max, loop_p->Attr("max"));
      step = d->AsDoc<ExprDoc>(loop->step, loop_p->Attr("step"));
      Array<ExprDoc> args;
      Array<StringRef> kwargs_keys;
      Array<ExprDoc> kwargs_values;
      if (min.defined()) {
        args.push_back(min.value());
      }
      if (max.defined()) {
        args.push_back(max.value());
      }
      if (step.defined()) {
        args.push_back(step.value());
      }
      ExprDoc rhs = IdDoc("range")->Call(args, kwargs_keys, kwargs_values);
      AsDocBody(loop->body, loop_p->Attr("body"), (*f).get(), d);
      return ForDoc(lhs, rhs, (*f)->stmts);
    });

// AutoFor
const char* AutoFor::TEMP_VALUE_VAR_KEY = "value_var";
const char* AutoFor::TEMP_ENUMERATE_POS_VAR_KEY = "enumerate_pos_var";
AutoFor::AutoFor(Array<BaseExpr> loop_vars, BaseExpr container, Stmt body, Span span) {
  MXCHECK(loop_vars.defined() && !loop_vars.empty());
  MXCHECK(container.defined());
  MXCHECK(body.defined());

  auto gen_var_name = [](const StringRef& prefix, const StringRef& seed, int i) -> StringRef {
    uint16_t hash_val = static_cast<uint16_t>(std::hash<StringRef>()(prefix + seed));
    StringRef ret =
        StringRef("__" + prefix + "_" + seed + std::to_string(hash_val) + "_" + std::to_string(i));
    return ret;
  };

  bool value_is_std_tuple = false;
  if (container.as<HLOEnumerateNode>() || container.as<HLOZipNode>()) {
    value_is_std_tuple = true;
  }

  ObjectPtr<AutoForNode> node = make_object<AutoForNode>();
  String temp_name;
  Array<Type> loop_var_types;
  for (auto i = 0; i < loop_vars.size(); ++i) {
    if (i > 0) {
      temp_name += "_";
    }
    if (auto prim_node = loop_vars[i].as<PrimVarNode>()) {
      temp_name += prim_node->name_hint;
    } else {
      auto loop_var_hlo = Downcast<HLOVar>(loop_vars[i]);
      temp_name += loop_var_hlo->name_hint();
    }
    loop_var_types.push_back(loop_vars[i]->checked_type());
  }

  // unroll zip container
  bool unroll_zip_state = false;
  if (auto* cons_ptr = container.as<HLOZipNode>()) {
    if (cons_ptr->values.size() == loop_vars.size()) {
      unroll_zip_state = true;
      // eval zip args
      for (auto i = 0; i < cons_ptr->values.size(); ++i) {
        String zip_arg_i_name = gen_var_name("reserved_eval_zip_arg", temp_name, i);
        auto eval_var = HLOVar(zip_arg_i_name, cons_ptr->values[i]->checked_type());
        node->eval_containers.push_back(eval_var);
      }
    }
  }
  // unroll enumerate container
  bool unroll_enumerate_state = false;
  if (auto* cons_ptr = container.as<HLOEnumerateNode>()) {
    if (2 == loop_vars.size()) {
      unroll_enumerate_state = true;
      // eval enumerate args
      String enumerate_arg_name = gen_var_name("reserved_eval_enumerate_arg", temp_name, 0);
      auto eval_var = HLOVar(enumerate_arg_name, cons_ptr->value->checked_type());
      node->eval_containers.push_back(eval_var);
    }
  }
  if (!unroll_zip_state && !unroll_enumerate_state) {
    String temp_cons_var_name = gen_var_name("reserved_eval_cons", temp_name, 0);
    auto eval_var = HLOVar(temp_cons_var_name, container->checked_type());
    node->eval_containers.push_back(eval_var);
  }

  // cache iter vars
  auto FuncCacheIterVar = [&](const BaseExpr& current_container, int index) {
    IteratorType iter_var_type(current_container->checked_type());
    String temp_iter_var_name = gen_var_name("reserved_iter", temp_name, index);
    auto iter_var = HLOVar(temp_iter_var_name, iter_var_type);
    node->iter_vars.push_back(iter_var);
    if (current_container->checked_type()->HasBeginEnd()) {
      // cache iter_end
      String temp_iter_end_var_name = gen_var_name("reserved_iter_end", temp_name, index);
      auto iter_end_var = HLOVar(temp_iter_end_var_name, iter_var_type);
      node->iter_end_vars.push_back(iter_end_var);
    } else {
      // cache has_next
      String has_next_var_name = gen_var_name("reserved_has_next", temp_name, index);
      auto has_next_var = PrimVar(has_next_var_name, runtime::DataType::Bool());
      node->iter_end_vars.push_back(has_next_var);
      // cache next_var_holder
      String next_holder_var_name = gen_var_name("reserved_next_holder", temp_name, index);
      auto next_holder_var = HLOVar(next_holder_var_name, ObjectType(false));
      node->loop_vars_holder.push_back(next_holder_var);
    }
  };
  if (unroll_zip_state) {
    auto* cons_ptr = container.as<HLOZipNode>();
    for (auto i = 0; i < cons_ptr->values.size(); ++i) {
      FuncCacheIterVar(cons_ptr->values[i], i);
    }
  } else if (unroll_enumerate_state) {
    auto* cons_ptr = container.as<HLOEnumerateNode>();
    FuncCacheIterVar(cons_ptr->value, 0);
    // cache pos
    String pos_var_name = gen_var_name("reserved_enum_pos", temp_name, 0);
    auto pos_var = PrimVar(pos_var_name, runtime::DataType::Int(64));
    node->temp_vars.Set(TEMP_ENUMERATE_POS_VAR_KEY, pos_var);
  } else {
    FuncCacheIterVar(container, 0);
    // cache loop vars
    if (loop_vars.size() > 1) {
      // cache value
      String temp_value_var_name = gen_var_name("reserved_value_tup", temp_name, 0);
      Type temp_value_var_type;
      if (value_is_std_tuple) {
        temp_value_var_type = TupleType(std::move(loop_var_types), value_is_std_tuple);
      } else {
        temp_value_var_type = InferIteratorValueType(container->checked_type());
      }
      auto value_var = HLOVar(temp_value_var_name, std::move(temp_value_var_type));
      node->temp_vars.Set(TEMP_VALUE_VAR_KEY, value_var);
    }
  }

  node->loop_vars = std::move(loop_vars);
  node->raw_container = std::move(container);
  node->body = std::move(body);
  node->span = std::move(span);

  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.AutoFor")
    .set_body_typed([](Array<BaseExpr> loop_var,
                       BaseExpr container,
                       Stmt body,
                       Span span = Span()) {
      return AutoFor(std::move(loop_var), std::move(container), std::move(body), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(AutoForNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::AutoFor>("", [](ir::AutoFor loop, ObjectPath loop_p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, loop);
      ExprDoc lhs{nullptr};
      if (loop->loop_vars.size() > 1) {
        int n = loop->loop_vars.size();
        Array<ExprDoc> loop_vars;
        loop_vars.reserve(n);
        for (int i = 0; i < n; ++i) {
          loop_vars.push_back(DefineVar(loop->loop_vars[i], *f, d));
        }
        lhs = TupleDoc(loop_vars);
      } else {
        lhs = DefineVar(loop->loop_vars[0], *f, d);
      }

      ExprDoc rhs = d->AsDoc<ExprDoc>(loop->raw_container, loop_p->Attr("raw_container"));
      AsDocBody(loop->body, loop_p->Attr("body"), (*f).get(), d);
      return ForDoc(lhs, rhs, (*f)->stmts);
    });

// While
While::While(BaseExpr cond, Stmt body, Span span) {
  MXCHECK(cond.defined());
  MXCHECK(body.defined());
  if (auto* cond_node = cond.as<PrimExprNode>()) {
    MXCHECK(cond_node->dtype.is_scalar());
  } else {
    auto hlo_cond = Downcast<HLOExpr>(cond);
    MXCHECK(hlo_cond->checked_type().defined());
    auto cond_type_node = hlo_cond->checked_type().as<PrimTypeNode>();
    MXCHECK(cond_type_node != nullptr && cond_type_node->dtype.is_scalar())
        << "[ir.While] cond is not Prim scalar";
  }

  ObjectPtr<WhileNode> node = make_object<WhileNode>();
  node->cond = std::move(cond);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.While")
    .set_body_typed([](BaseExpr cond, Stmt body, Span span = Span()) {
      return While(cond, body, span);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(WhileNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::While>("", [](ir::While loop, ObjectPath loop_p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, loop);
      ExprDoc cond = d->AsDoc<ExprDoc>(loop->cond, loop_p->Attr("cond"));
      AsDocBody(loop->body, loop_p->Attr("body"), (*f).get(), d);
      return WhileDoc(cond, (*f)->stmts);
    });

// Break
Break::Break() {
  data_ = make_object<BreakNode>();
}

MATXSCRIPT_REGISTER_GLOBAL("ir.Break").set_body_typed([]() { return Break(); });
MATXSCRIPT_REGISTER_NODE_TYPE(BreakNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::Break>("", [](ir::Break n, ObjectPath p, IRDocsifier d) -> Doc {
      return BreakDoc();
    });

// Continue
Continue::Continue() {
  data_ = make_object<ContinueNode>();
}

MATXSCRIPT_REGISTER_GLOBAL("ir.Continue").set_body_typed([]() { return Continue(); });
MATXSCRIPT_REGISTER_NODE_TYPE(ContinueNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::Continue>("", [](ir::Continue n, ObjectPath p, IRDocsifier d) -> Doc {
      return ContinueDoc();
    });

// SeqStmt
SeqStmt::SeqStmt(Array<Stmt> seq, Span span) {
  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.SeqStmt").set_body_typed([](Array<Stmt> seq, Span span = Span()) {
  return SeqStmt(std::move(seq), std::move(span));
});

MATXSCRIPT_REGISTER_NODE_TYPE(SeqStmtNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::SeqStmt>("", [](ir::SeqStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, stmt);
      AsDocBody(stmt, p, f->get(), d);
      return StmtBlockDoc((*f)->stmts);
    });

// IfThenElse
IfThenElse::IfThenElse(BaseExpr condition, Stmt then_case, Stmt else_case, Span span) {
  MXCHECK(condition.defined());
  MXCHECK(then_case.defined());
  // else_case may be null.
  ObjectPtr<IfThenElseNode> node = make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(IfThenElseNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.IfThenElse")
    .set_body_typed([](BaseExpr condition, Stmt then_case, Stmt else_case, Span span = Span()) {
      return IfThenElse(condition, then_case, else_case, span);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::IfThenElse>(  //
        "",
        [](ir::IfThenElse stmt, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          Array<StmtDoc> then_branch;
          Array<StmtDoc> else_branch;
          if (stmt->then_case.defined()) {
            With<IRFrame> f(d, stmt->then_case);
            AsDocBody(stmt->then_case, p->Attr("then_case"), f->get(), d);
            then_branch = (*f)->stmts;
          }
          if (stmt->else_case.defined()) {
            With<IRFrame> f(d, stmt->else_case);
            AsDocBody(stmt->else_case, p->Attr("else_case"), f->get(), d);
            else_branch = (*f)->stmts;
          }
          return IfDoc(cond, then_branch, else_branch);
        });

// ExceptionHandler
ExceptionHandler::ExceptionHandler(BaseExpr e, Stmt body, Span span) {
  MXCHECK(body.defined()) << "body is not defined!!!";
  MXCHECK(!e.defined()) << "specific exception is not supported now!!!";
  ObjectPtr<ExceptionHandlerNode> node = make_object<ExceptionHandlerNode>();
  node->body = std::move(body);
  node->e = std::move(e);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ExceptionHandlerNode);
MATXSCRIPT_REGISTER_GLOBAL("ir.ExceptionHandler")
    .set_body_typed([](BaseExpr e, Stmt body, Span span = Span()) {
      return ExceptionHandler(std::move(e), std::move(body), std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::ExceptionHandler>(  //
        "",
        [](ir::ExceptionHandler stmt, ObjectPath p, IRDocsifier d) -> Doc {
          MXCHECK(!stmt->e.defined()) << "specific exception is not supported now!!!";
          Array<StmtDoc> body_branch;
          if (stmt->body.defined()) {
            With<IRFrame> f(d, stmt->body);
            AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
            body_branch = (*f)->stmts;
          }
          return ExceptionHandlerDoc(NullOpt, NullOpt, body_branch);
        });

// TryExcept
TryExcept::TryExcept(Stmt body, Array<ExceptionHandler> handlers, Span span) {
  MXCHECK(body.defined()) << "body is not defined!!!";
  MXCHECK(handlers.defined() && handlers.size() == 1)
      << "only one except handler is supported now!!!";
  ObjectPtr<TryExceptNode> node = make_object<TryExceptNode>();
  node->body = std::move(body);
  node->handlers = std::move(handlers);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(TryExceptNode);
MATXSCRIPT_REGISTER_GLOBAL("ir.TryExcept")
    .set_body_typed([](Stmt body, Array<ExceptionHandler> handlers, Span span = Span()) {
      return TryExcept(std::move(body), std::move(handlers), std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::TryExcept>(  //
        "",
        [](ir::TryExcept stmt, ObjectPath p, IRDocsifier d) -> Doc {
          Array<StmtDoc> body;
          if (stmt->body.defined()) {
            With<IRFrame> f(d, stmt->body);
            AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
            body = (*f)->stmts;
          }
          Array<ExceptionHandlerDoc> handlers;
          auto p_handlers = p->Attr("handlers");
          int n = stmt->handlers.size();
          handlers.reserve(n);
          for (int i = 0; i < n; ++i) {
            auto doc = d->AsDoc<ExceptionHandlerDoc>(stmt->handlers[i], p_handlers->ArrayIndex(i));
            handlers.push_back(std::move(doc));
          }
          return TryExceptDoc(body, handlers, NullOpt, NullOpt);
        });

// Raise
Raise::Raise(BaseExpr exc, Span span) {
  ObjectPtr<RaiseNode> node = make_object<RaiseNode>();
  node->exc = std::move(exc);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(RaiseNode);
MATXSCRIPT_REGISTER_GLOBAL("ir.Raise").set_body_typed([](BaseExpr exc, Span span = Span()) {
  return Raise(std::move(exc), std::move(span));
});

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::Raise>(  //
        "",
        [](ir::Raise stmt, ObjectPath p, IRDocsifier d) -> Doc {
          if (stmt->exc.defined()) {
            ExprDoc exc = d->AsDoc<ExprDoc>(stmt->exc, p->Attr("exc"));
            return RaiseDoc(exc, NullOpt);
          } else {
            return RaiseDoc(NullOpt, NullOpt);
          }
        });

// HLOYield
HLOYield::HLOYield(BaseExpr symbol, BaseExpr label, Span span) {
  ObjectPtr<HLOYieldNode> n = make_object<HLOYieldNode>();
  n->symbol = std::move(symbol);
  n->label = std::move(label);
  n->span = std::move(span);
  data_ = std::move(n);
}

HLOYield::HLOYield(BaseExpr symbol, Span span)
    : HLOYield(std::move(symbol), IntImm(runtime::DataType::Int(64), 0), std::move(span)){};

MATXSCRIPT_REGISTER_NODE_TYPE(HLOYieldNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.HLOYield").set_body_typed([](BaseExpr symbol, Span span = Span()) {
  return HLOYield(std::move(symbol), span);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::HLOYield>(  //
        "",
        [](ir::HLOYield stmt, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc value = d->AsDoc<ExprDoc>(stmt->symbol, p->Attr("symbol"));
          return ExprStmtDoc(YieldDoc(value));
        });

}  // namespace ir
}  // namespace matxscript
