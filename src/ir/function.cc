// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the expressions is inspired by TVM.
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
 * \file src/ir/function.cc
 * \brief The function data structure.
 */
#include <matxscript/ir/function.h>

#include <matxscript/ir/_base/with.h>
#include <matxscript/ir/hlo_builtin.h>
#include <matxscript/ir/prim_ops.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/ir/printer/utils.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;
using namespace ::matxscript::ir::printer;

bool BaseFuncNode::HasGlobalName() const {
  auto global_symbol = GetAttr<StringRef>(attr::kGlobalSymbol);
  return global_symbol.defined();
}

StringRef BaseFuncNode::GetGlobalName() const {
  auto global_symbol = GetAttr<StringRef>(attr::kGlobalSymbol);
  MXCHECK(global_symbol.defined()) << "Expect BaseFunc to have the global_symbol attribute";
  return global_symbol.value();
}

bool BaseFuncNode::HasBoundName() const {
  auto global_symbol = GetAttr<StringRef>(attr::kBoundSymbol);
  return global_symbol.defined();
}

StringRef BaseFuncNode::GetBoundName() const {
  auto global_symbol = GetAttr<StringRef>(attr::kBoundSymbol);
  MXCHECK(global_symbol.defined()) << "Expect BaseFunc to have the bound_symbol attribute";
  return global_symbol.value();
}

bool BaseFuncNode::ExportSymbol() const {
  auto export_symbol = GetAttr<Bool>(attr::kExportSymbol, Bool(false));
  return export_symbol.value().operator bool();
}

bool BaseFuncNode::CaptureSessionHandle() const {
  auto value = GetAttr<Bool>(attr::kCaptureSessionHandle, Bool(false));
  return value.value().operator bool();
}

bool BaseFuncNode::IsClassConstructor() const {
  auto is_cons = GetAttr<Bool>(attr::kClassConstructor, Bool(false));
  return is_cons.value().operator bool();
}

bool BaseFuncNode::IsClassMember() const {
  auto val = GetAttr<StringRef>(attr::kClassNameBelongTo);
  return val.operator bool();
}

StringRef BaseFuncNode::GetBelongToClassName() const {
  auto val = GetAttr<StringRef>(attr::kClassNameBelongTo, "");
  return val.value();
}

/******************************************************************************
 * PrimFunc
 *****************************************************************************/

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(Array<PrimVar> params,
                   Array<PrimExpr> default_params,
                   Stmt body,
                   Type ret_type,
                   DictAttrs attrs,
                   Span span) {
  // Assume void-return type for now
  // TODO(tvm-team) consider type deduction from body.
  if (!ret_type.defined()) {
    ret_type = VoidType();
  }
  auto n = make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->default_params = std::move(default_params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

FuncType PrimFuncNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(GetType(runtime::Downcast<PrimExpr>(param)));
  }
  return FuncType(param_types, ret_type, {}, {});
}

Array<BaseExpr> PrimFuncNode::GetParams() const {
  Array<BaseExpr> result;
  for (auto& param : params) {
    result.push_back(param);
  }
  return result;
}

Array<BaseExpr> PrimFuncNode::GetDefaultParams() const {
  return Downcast<Array<BaseExpr>>(default_params);
}

Type PrimFuncNode::GetReturnType() const {
  return ret_type;
}

Stmt PrimFuncNode::GetBody() const {
  return body;
}

StringRef PrimFuncNode::GetReprName() const {
  return "primfn";
}

MATXSCRIPT_REGISTER_NODE_TYPE(PrimFuncNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& ref, ReprPrinter* p) {
      // TODO(tvm-team) redirect to Text printer once we have a good text format.
      auto* node = static_cast<const PrimFuncNode*>(ref.get());
      p->stream << "PrimFunc(" << node->params << ") ";
      if (node->attrs.defined()) {
        p->stream << "attrs=" << node->attrs;
      }
      p->stream << " {\n";
      p->indent += 2;
      p->Print(node->body);
      p->indent -= 2;
      p->stream << "}\n";
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimFunc")
    .set_body_typed([](Array<PrimVar> params,
                       Array<PrimExpr> default_params,
                       Stmt body,
                       Type ret_type,
                       DictAttrs attrs) {
      return PrimFunc(std::move(params),
                      std::move(default_params),
                      std::move(body),
                      std::move(ret_type),
                      std::move(attrs));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::PrimFunc>("", [](ir::PrimFunc func, ObjectPath p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, func);
      (*f)->AddDispatchToken(d, "ir");
      d->SetCommonPrefix(func, [](const ObjectRef& obj) {
        return obj->IsInstance<ir::PrimVarNode>() || obj->IsInstance<ir::BufferNode>();
      });
      int n_args = func->params.size();
      // Step 1. Handle `func->params`
      int default_begin_pos = func->params.size() - func->default_params.size();
      Array<AssignDoc> args;
      args.reserve(n_args);
      for (int i = 0; i < n_args; ++i) {
        ir::PrimVar var = func->params[i];
        ObjectPath var_p = p->Attr("params")->ArrayIndex(i);
        ExprDoc a = d->AsDoc<ExprDoc>(var->type_annotation, var_p->Attr("type_annotation"));
        Optional<ExprDoc> rhs = NullOpt;
        if (i >= default_begin_pos) {
          int def_pos = i - default_begin_pos;
          rhs = d->AsDoc<ExprDoc>(func->default_params[def_pos],
                                  p->Attr("default_params")->ArrayIndex(def_pos));
        }
        args.push_back(AssignDoc(DefineVar(var, *f, d), rhs, a));
      }
      // Step 2. Handle `func->attrs`
      if (func->attrs.defined() && !func->attrs->dict.empty()) {
        (*f)->stmts.push_back(
            ExprStmtDoc(Dialect(d, "func_attr")  //
                            ->Call({d->AsDoc<ExprDoc>(func->attrs, p->Attr("attrs"))})));
      }
      // Step 3. Handle `func->body`
      AsDocBody(func->body, p->Attr("body"), f->get(), d);
      Optional<ExprDoc> ret_type = NullOpt;
      ret_type = d->AsDoc<ExprDoc>(func->ret_type, p->Attr("ret_type"));
      StringRef fn_name = func->HasGlobalName() ? func->GetGlobalName() : "main";
      return FunctionDoc(
          /*name=*/IdDoc(fn_name),
          /*args=*/args,
          /*decorators=*/{Dialect(d, "kernel")},
          /*return_type=*/ret_type,
          /*body=*/(*f)->stmts);
    });

/******************************************************************************
 * Function
 *****************************************************************************/

Function::Function(Array<BaseExpr> params,
                   Array<BaseExpr> default_params,
                   Stmt body,
                   Type ret_type,
                   Array<TypeVar> type_params,
                   DictAttrs attrs,
                   Span span) {
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  MXCHECK(params.defined());
  MXCHECK(type_params.defined());
  // TODO(mxiandi) : check params is Var or PrimVar
  n->params = std::move(params);
  n->default_params = std::move(default_params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

FuncType FunctionNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto& param : this->params) {
    Type param_type;
    if (auto* prim_var = param.as<PrimVarNode>()) {
      param_type = PrimType(prim_var->dtype);
    } else if (auto* hlo_var = param.as<HLOVarNode>()) {
      param_type = hlo_var->type_annotation;
    } else {
      MXCHECK(false) << "Function's param is not a PrimVar or HLOVar";
    }
    param_types.push_back(param_type);
  }

  Type ret_type = this->ret_type;
  return FuncType(param_types, ret_type, this->type_params, {});
}

Array<BaseExpr> FunctionNode::GetParams() const {
  return params;
}

Array<BaseExpr> FunctionNode::GetDefaultParams() const {
  return default_params;
}

Type FunctionNode::GetReturnType() const {
  return ret_type;
}

Stmt FunctionNode::GetBody() const {
  return body;
}

StringRef FunctionNode::GetReprName() const {
  return "fn";
}

MATXSCRIPT_REGISTER_NODE_TYPE(FunctionNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.Function")
    .set_body_typed([](Array<BaseExpr> params,
                       Array<BaseExpr> default_params,
                       Stmt body,
                       Type ret_type,
                       Array<TypeVar> ty_params,
                       DictAttrs attrs,
                       Span span) {
      return Function(std::move(params),
                      std::move(default_params),
                      std::move(body),
                      std::move(ret_type),
                      std::move(ty_params),
                      std::move(attrs),
                      std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FunctionNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FunctionNode*>(ref.get());
      p->stream << "FunctionNode(" << node->params << ", " << node->ret_type << ", " << node->body
                << ", " << node->type_params << ", " << node->attrs << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::Function>("", [](ir::Function func, ObjectPath p, IRDocsifier d) -> Doc {
      With<IRFrame> f(d, func);
      (*f)->AddDispatchToken(d, "ir");
      d->SetCommonPrefix(func, [](const ObjectRef& obj) {
        return obj->IsInstance<ir::PrimVarNode>() || obj->IsInstance<ir::HLOVarNode>() ||
               obj->IsInstance<ir::BufferNode>();
      });
      int n_args = func->params.size();
      // Step 1. Handle `func->params`
      int default_begin_pos = func->params.size() - func->default_params.size();
      Array<AssignDoc> args;
      args.reserve(n_args);
      bool is_method = func->IsClassMember();
      for (int i = 0; i < n_args; ++i) {
        ir::BaseExpr var = func->params[i];
        ObjectPath var_p = p->Attr("params")->ArrayIndex(i);
        ExprDoc a = d->AsDoc<ExprDoc>(var->checked_type_, var_p->Attr("checked_type_"));
        Optional<ExprDoc> rhs = NullOpt;
        if (i >= default_begin_pos) {
          int def_pos = i - default_begin_pos;
          rhs = d->AsDoc<ExprDoc>(func->default_params[def_pos],
                                  p->Attr("default_params")->ArrayIndex(def_pos));
        }
        args.push_back(AssignDoc(DefineVar(var, *f, d), rhs, a));
      }
      // Step 2. Handle `class->attrs`
      if (func->attrs.defined() && !func->attrs->dict.empty()) {
        auto attrs = IRTextPrinter::Print(func->attrs, d->cfg);
        (*f)->stmts.push_back(CommentDoc(attrs));
      }
      // Step 3. Handle `func->body`
      AsDocBody(func->body, p->Attr("body"), f->get(), d);
      Optional<ExprDoc> ret_type = NullOpt;
      ret_type = d->AsDoc<ExprDoc>(func->ret_type, p->Attr("ret_type"));

      StringRef fn_name = is_method ? (func->HasBoundName() ? func->GetBoundName() : "main")
                                    : (func->HasGlobalName() ? func->GetGlobalName() : "main");
      Array<ExprDoc> decorators;
      if (!is_method) {
        decorators.push_back(Dialect(d, "script"));
      }
      return FunctionDoc(
          /*name=*/IdDoc(fn_name),
          /*args=*/args,
          /*decorators=*/decorators,
          /*return_type=*/ret_type,
          /*body=*/(*f)->stmts);
    });

/******************************************************************************
 * LambdaFunction
 *****************************************************************************/

LambdaFunction::LambdaFunction(
    Array<BaseExpr> captures, Array<BaseExpr> params, Stmt body, Type ret_type, Span span) {
  ObjectPtr<LambdaFunctionNode> n = make_object<LambdaFunctionNode>();
  MXCHECK(params.defined());
  // TODO(mxiandi) : check params is Var or PrimVar
  n->captures = std::move(captures);
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(LambdaFunctionNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.LambdaFunction")
    .set_body_typed(
        [](Array<BaseExpr> captures, Array<BaseExpr> params, Stmt body, Type ret_type, Span span) {
          return LambdaFunction(std::move(captures),
                                std::move(params),
                                std::move(body),
                                std::move(ret_type),
                                std::move(span));
        });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LambdaFunctionNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const LambdaFunctionNode*>(ref.get());
      p->stream << "LambdaFunctionNode(" << node->params << ", " << node->ret_type << ", "
                << node->body << ", " << node->captures << ")";
    });

namespace {
struct LambdaFunctionToComprehension {
  struct ComprehensionHelper {
    ExprDoc target{nullptr};
    ExprDoc iter{nullptr};
    Array<ExprDoc> ifs;
  };

  void VisitExpr_(const CallNode* op, ObjectPath p) {
    if (op->op.same_as(builtin::list_append()) || op->op.same_as(builtin::ft_list_append())) {
      elt.push_back(d->AsDoc<ExprDoc>(op->args[1], p->Attr("args")->ArrayIndex(1)));
    } else if (op->op.same_as(builtin::set_add()) || op->op.same_as(builtin::ft_set_add())) {
      elt.push_back(d->AsDoc<ExprDoc>(op->args[1], p->Attr("args")->ArrayIndex(1)));
    } else if (op->op.same_as(builtin::dict___setitem__()) ||
               op->op.same_as(builtin::ft_dict___setitem__())) {
      elt.push_back(d->AsDoc<ExprDoc>(op->args[1], p->Attr("args")->ArrayIndex(1)));
      elt.push_back(d->AsDoc<ExprDoc>(op->args[2], p->Attr("args")->ArrayIndex(2)));
    } else {
      MXTHROW << "internal error: ";
    }
  }

  void VisitStmt_(const IfThenElseNode* op, ObjectPath p) {
    ExprDoc cond = d->AsDoc<ExprDoc>(op->condition, p->Attr("condition"));
    MXCHECK(!op->else_case.defined()) << "invalid syntax";
    docs.back().ifs.push_back(cond);
    this->Visit(op->then_case, p->Attr("then_case"));
  }

  void VisitStmt_(const ForNode* op, ObjectPath p) {
    ExprDoc target = d->AsDoc<ExprDoc>(op->loop_var, p->Attr("loop_var"));
    Optional<ExprDoc> min = NullOpt;
    Optional<ExprDoc> max = NullOpt;
    Optional<ExprDoc> step = NullOpt;
    min = d->AsDoc<ExprDoc>(op->min, p->Attr("min"));
    max = d->AsDoc<ExprDoc>(op->max, p->Attr("max"));
    step = d->AsDoc<ExprDoc>(op->step, p->Attr("step"));
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
    Array<ExprDoc> ifs;
    docs.push_back(ComprehensionHelper{target, rhs, ifs});
    this->Visit(op->body, p->Attr("body"));
  }

  void VisitStmt_(const AutoForNode* op, ObjectPath p) {
    ExprDoc lhs{nullptr};
    if (op->loop_vars.size() > 1) {
      int n = op->loop_vars.size();
      Array<ExprDoc> loop_vars;
      loop_vars.reserve(n);
      for (int i = 0; i < n; ++i) {
        loop_vars.push_back(
            d->AsDoc<ExprDoc>(op->loop_vars[i], p->Attr("loop_vars")->ArrayIndex(i)));
      }
      lhs = TupleDoc(loop_vars);
    } else {
      lhs = d->AsDoc<ExprDoc>(op->loop_vars[0], p->Attr("loop_vars")->ArrayIndex(0));
    }
    ExprDoc rhs = d->AsDoc<ExprDoc>(op->raw_container, p->Attr("raw_container"));
    Array<ExprDoc> ifs;
    docs.push_back(ComprehensionHelper{lhs, rhs, ifs});
    this->Visit(op->body, p->Attr("body"));
  }

  void Visit(ObjectRef op, ObjectPath p) {
    if (auto const* node = op.as<AutoForNode>()) {
      VisitStmt_(node, p);
    } else if (auto const* node = op.as<ForNode>()) {
      VisitStmt_(node, p);
    } else if (auto const* node = op.as<IfThenElseNode>()) {
      VisitStmt_(node, p);
    } else if (auto const* node = op.as<CallNode>()) {
      VisitExpr_(node, p);
    } else if (auto const* node = op.as<ExprStmtNode>()) {
      this->Visit(node->expr, p->Attr("expr"));
    } else {
      MXTHROW << "internal error: " << op->GetTypeKey();
    }
  }

  Array<ComprehensionDoc> GenCompList() const {
    Array<ComprehensionDoc> comps;
    comps.reserve(int64_t(docs.size()));
    for (auto& ch : docs) {
      comps.push_back(ComprehensionDoc(ch.target, ch.iter, ch.ifs));
    }
    return comps;
  }

  std::vector<ExprDoc> elt;
  std::vector<ComprehensionHelper> docs;
  IRDocsifier d{nullptr};
};

}  // namespace

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::LambdaFunction>(
        "", [](ir::LambdaFunction func, ObjectPath p, IRDocsifier d) -> Doc {
          // Currently Lambda functions are mainly used to represent Comprehension.
          With<IRFrame> f(d, func);
          (*f)->AddDispatchToken(d, "ir");
          d->SetCommonPrefix(func, [](const ObjectRef& obj) {
            return obj->IsInstance<ir::PrimVarNode>() || obj->IsInstance<ir::HLOVarNode>() ||
                   obj->IsInstance<ir::BufferNode>();
          });
          // Step 1. Handle `func->params`
          MXCHECK(func->params.empty()) << "internal error";
          // Step 2. Handle `func->body`
          MXCHECK(func->body->IsInstance<SeqStmtNode>()) << "internal error";
          const auto* body_node = func->body.as<SeqStmtNode>();
          ObjectRef for_n{nullptr};
          int count = 0;
          int i = 0;
          for (; i < body_node->size(); ++i) {
            if (body_node->seq[i].as<ForNode>() || body_node->seq[i].as<AutoForNode>()) {
              for_n = body_node->seq[i];
              count++;
            }
          }
          MXCHECK(count == 1 && for_n.defined()) << "internal error";
          LambdaFunctionToComprehension helper;
          helper.d = d;
          helper.Visit(for_n, p->Attr("body")->ArrayIndex(i));

          if (func->ret_type->IsInstance<ListTypeNode>()) {
            MXCHECK(helper.elt.size() == 1) << "internal error";
            return ListCompDoc(helper.elt[0], helper.GenCompList());
          } else if (func->ret_type->IsInstance<SetTypeNode>()) {
            MXCHECK(helper.elt.size() == 1) << "internal error";
            return SetCompDoc(helper.elt[0], helper.GenCompList());
          } else if (func->ret_type->IsInstance<DictTypeNode>()) {
            MXCHECK(helper.elt.size() == 2) << "internal error";
            return DictCompDoc(helper.elt[0], helper.elt[1], helper.GenCompList());
          } else {
            MXTHROW << "unexpected lambda function ret_type: " << func->ret_type;
          }
          return ExprDoc{nullptr};
        });

/******************************************************************************
 * BaseFunc
 *****************************************************************************/

MATXSCRIPT_REGISTER_GLOBAL("ir.BaseFunc_Attrs").set_body_typed([](BaseFunc func) {
  return func->attrs;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.BaseFuncCopy").set_body_typed([](BaseFunc func) { return func; });

MATXSCRIPT_REGISTER_GLOBAL("ir.BaseFuncWithAttr")
    .set_body_typed([](BaseFunc func, StringRef key, RTValue arg_val) -> BaseFunc {
      ObjectRef value = StringRef::CanConvertFrom(arg_val) ? arg_val.As<StringRef>()
                                                           : arg_val.AsObjectRef<ObjectRef>();
      if (func->IsInstance<PrimFuncNode>()) {
        return WithAttr(runtime::Downcast<PrimFunc>(std::move(func)), std::move(key), value);
      } else if (func->IsInstance<FunctionNode>()) {
        return WithAttr(runtime::Downcast<Function>(std::move(func)), std::move(key), value);
      } else {
        MXTHROW << "Do not support function type " << func->GetTypeKey();
        return func;
      }
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.BaseFunc_GetFuncType").set_body_typed([](BaseFunc func) {
  return func->func_type_annotation();
});

}  // namespace ir
}  // namespace matxscript
