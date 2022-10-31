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

#include <matxscript/ir/prim_ops.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;

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
  n->checked_type_ = n->func_type_annotation();
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

runtime::Array<BaseExpr> PrimFuncNode::GetParams() const {
  runtime::Array<BaseExpr> result;
  for (auto& param : params) {
    result.push_back(param);
  }
  return result;
}

runtime::Array<BaseExpr> PrimFuncNode::GetDefaultParams() const {
  return Downcast<runtime::Array<BaseExpr>>(default_params);
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
  n->checked_type_ = n->ret_type;
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

runtime::Array<BaseExpr> FunctionNode::GetParams() const {
  return params;
}

runtime::Array<BaseExpr> FunctionNode::GetDefaultParams() const {
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

/******************************************************************************
 * LambdaFunction
 *****************************************************************************/

LambdaFunction::LambdaFunction(Array<BaseExpr> captures,
                               Array<BaseExpr> params,
                               Stmt body,
                               Type ret_type,
                               DictAttrs attrs,
                               Span span) {
  ObjectPtr<LambdaFunctionNode> n = make_object<LambdaFunctionNode>();
  MXCHECK(params.defined());
  // TODO(mxiandi) : check params is Var or PrimVar
  n->captures = std::move(captures);
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->checked_type_ = n->ret_type;
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

FuncType LambdaFunctionNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto& param : this->params) {
    Type param_type;
    if (auto* prim_var = param.as<PrimVarNode>()) {
      param_type = PrimType(prim_var->dtype);
    } else if (auto* hlo_var = param.as<HLOVarNode>()) {
      param_type = hlo_var->type_annotation;
    } else {
      MXCHECK(false) << "LambdaFunction's param is not a PrimVar or HLOVar";
    }
    param_types.push_back(param_type);
  }

  Type ret_type = this->ret_type;
  return FuncType(param_types, ret_type, {}, {});
}

runtime::Array<BaseExpr> LambdaFunctionNode::GetParams() const {
  return params;
}

runtime::Array<BaseExpr> LambdaFunctionNode::GetDefaultParams() const {
  return {};
}

Type LambdaFunctionNode::GetReturnType() const {
  return ret_type;
}

Stmt LambdaFunctionNode::GetBody() const {
  return body;
}

StringRef LambdaFunctionNode::GetReprName() const {
  return "lambda";
}

MATXSCRIPT_REGISTER_NODE_TYPE(LambdaFunctionNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.LambdaFunction")
    .set_body_typed([](Array<BaseExpr> captures,
                       Array<BaseExpr> params,
                       Stmt body,
                       Type ret_type,
                       DictAttrs attrs,
                       Span span) {
      return LambdaFunction(std::move(captures),
                            std::move(params),
                            std::move(body),
                            std::move(ret_type),
                            std::move(attrs),
                            std::move(span));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LambdaFunctionNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const LambdaFunctionNode*>(ref.get());
      p->stream << "LambdaFunctionNode(" << node->params << ", " << node->ret_type << ", "
                << node->body << ", " << node->captures << ", " << node->attrs << ")";
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
      } else if (func->IsInstance<LambdaFunctionNode>()) {
        return WithAttr(runtime::Downcast<LambdaFunction>(std::move(func)), std::move(key), value);
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
