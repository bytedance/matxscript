// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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
#include "full_typed_optimizer.h"

#include <matxscript/ir/hlo_builtin.h>

namespace matxscript {
namespace ir {

const BaseExprNode* FullTypedOptimizerAnalysis::RemoveMove(const BaseExprNode* node) {
  if (node->IsInstance<HLOMoveNode>()) {
    return RemoveMove(static_cast<const HLOMoveNode*>(node)->value.get());
  }
  return node;
}

static const BaseExprNode* RemoveMoveAndCast(const BaseExprNode* node) {
  if (node->IsInstance<HLOMoveNode>()) {
    return RemoveMoveAndCast(static_cast<const HLOMoveNode*>(node)->value.get());
  }
  if (node->IsInstance<HLOCastNode>()) {
    return RemoveMoveAndCast(static_cast<const HLOCastNode*>(node)->value.get());
  }
  if (node->IsInstance<HLOCastPrimNode>()) {
    return RemoveMoveAndCast(static_cast<const HLOCastPrimNode*>(node)->value.get());
  }
  return node;
}

static bool NoNeedCheck(const Type& var_type, const CallNode* call) {
  return true;
}

template <size_t index, typename VarTypeNode>
static bool ListOrSetCheckNthTypeEqual(const Type& var_type, const CallNode* call) {
  if (index >= call->args.size()) {
    // default args
    return true;
  }
  const auto& var_type_2 = RemoveReference(var_type);
  if (auto* li_node = var_type_2.template as<VarTypeNode>()) {
    const auto& item_type = RemoveReference(li_node->item_type);
    const auto* arg_i_node = call->args[index].get();
    const auto& arg_i_origin_type = RemoveReference(arg_i_node->checked_type());

    arg_i_node = RemoveMoveAndCast(arg_i_node);
    const auto& arg_i_type = RemoveReference(arg_i_node->checked_type());
    return runtime::StructuralEqual()(arg_i_origin_type, item_type) ||
           runtime::StructuralEqual()(arg_i_type, item_type);
  }
  return false;
}

FullTypedOptimizerAnalysis::FullTypedOptimizerAnalysis()
    : StmtExprVisitor(),
      supported_list_ops{
          // ops no need check
          {builtin::list___len__().get(), NoNeedCheck},
          {builtin::list_reserve().get(), NoNeedCheck},
          {builtin::list___getitem__().get(), NoNeedCheck},
          {builtin::list_capacity().get(), NoNeedCheck},
          {builtin::list_pop().get(), NoNeedCheck},
          {builtin::list_clear().get(), NoNeedCheck},
          {builtin::list_reverse().get(), NoNeedCheck},
          {builtin::list_sort_no_key().get(), NoNeedCheck},
          {builtin::list_sort().get(), NoNeedCheck},

          // check argument type
          {builtin::list___contains__().get(), ListOrSetCheckNthTypeEqual<1, ListTypeNode>},
          {builtin::list___setitem__().get(), ListOrSetCheckNthTypeEqual<2, ListTypeNode>},
          {builtin::list_append().get(), ListOrSetCheckNthTypeEqual<1, ListTypeNode>},
          {builtin::list_index().get(), ListOrSetCheckNthTypeEqual<1, ListTypeNode>},
          {builtin::list_insert().get(), ListOrSetCheckNthTypeEqual<2, ListTypeNode>},
          {builtin::list_remove().get(), ListOrSetCheckNthTypeEqual<1, ListTypeNode>},
          {builtin::list_count().get(), ListOrSetCheckNthTypeEqual<1, ListTypeNode>},

      },
      supported_dict_ops{},
      supported_set_ops() {
}

bool FullTypedOptimizerAnalysis::IsListLiteral(const BaseExprNode* init) {
  if (init->IsInstance<InitializerListNode>()) {
    return true;
  }
  if (init->IsInstance<CallNode>()) {
    auto* call_node = static_cast<const CallNode*>(init);
    {
      auto* constructor_op = call_node->op.as<ConstructorNode>();
      if (constructor_op && call_node->args.size() == 0) {
        return true;
      }
      if (constructor_op && call_node->args.size() == 1) {
        return IsListLiteral(call_node->args[0].get());
      }
    }
    {
      if (call_node->op.same_as(builtin::list_fused_repeat_one()) && call_node->args.size() == 2) {
        return true;
      }
      if (call_node->op.same_as(builtin::list_fused_repeat_many()) && call_node->args.size() == 2) {
        return IsListLiteral(call_node->args[0].get());
      }
    }
  }
  return false;
}

bool FullTypedOptimizerAnalysis::IsDictLiteral(const BaseExprNode* init) {
  if (init->IsInstance<InitializerDictNode>()) {
    return true;
  }
  if (init->IsInstance<CallNode>()) {
    auto* call_node = static_cast<const CallNode*>(init);
    auto* call_op = call_node->op.as<ConstructorNode>();
    if (call_op && call_node->args.size() == 0) {
      return true;
    }
    if (call_op && call_node->args.size() == 1) {
      return IsDictLiteral(call_node->args[0].get());
    }
  }
  return false;
}

bool FullTypedOptimizerAnalysis::IsCandidate(const BaseExprNode* var, const BaseExprNode* init) {
  if (!var->IsInstance<HLOVarNode>()) {
    return false;
  }
  const auto& type = RemoveReference(var->checked_type());
  if (auto* type_node = type.as<ListTypeNode>()) {
    if (type_node->is_full_typed) {
      return false;
    }
    return IsListLiteral(init);
  }
  if (auto* type_node = type.as<SetTypeNode>()) {
    if (type_node->is_full_typed) {
      return false;
    }
    return IsListLiteral(init);
  }
  if (auto* type_node = type.as<DictTypeNode>()) {
    if (type_node->is_full_typed) {
      return false;
    }
    return IsDictLiteral(init);
  }
  return false;
}

static runtime::Array<BaseExpr> GetListLiteralValues(const BaseExprNode* init) {
  if (init->IsInstance<InitializerListNode>()) {
    return static_cast<const InitializerListNode*>(init)->fields;
  }
  MXCHECK(init->IsInstance<CallNode>());
  auto* call_node = static_cast<const CallNode*>(init);
  if (call_node->op.as<ConstructorNode>()) {
    if (call_node->args.size() == 0) {
      return {};
    }
    MXCHECK(call_node->args.size() == 1) << "argument size must be 0 or 1";
    return GetListLiteralValues(call_node->args[0].get());
  }
  if (call_node->op.same_as(builtin::list_fused_repeat_one())) {
    MXCHECK(call_node->args.size() == 2) << "internal error";
    return {call_node->args[1]};
  }
  if (call_node->op.same_as(builtin::list_fused_repeat_many())) {
    MXCHECK(call_node->args.size() == 2) << "internal error";
    return GetListLiteralValues(call_node->args[1].get());
  }
  MXTHROW << "internal error";
  return {};
}

static runtime::Map<BaseExpr, BaseExpr> GetDictLiteralValues(const BaseExprNode* init) {
  if (init->IsInstance<InitializerDictNode>()) {
    return static_cast<const InitializerDictNode*>(init)->fields;
  }
  MXCHECK(init->IsInstance<CallNode>());
  auto* call_node = static_cast<const CallNode*>(init);
  auto* call_op = call_node->op.as<ConstructorNode>();
  MXCHECK(call_op) << "[internal error] expect the op is an constructor";
  if (call_node->args.size() == 0) {
    return {};
  }
  MXCHECK(call_node->args.size() == 1) << "argument size must be 0 or 1";
  return GetDictLiteralValues(call_node->args[0].get());
}

Type FullTypedOptimizerAnalysis::InferNewVarType(const BaseExprNode* var,
                                                 const BaseExprNode* init) {
  const auto& type = RemoveReference(var->checked_type());
  if (auto* type_node = type.as<ListTypeNode>()) {
    const Type& item_type = type_node->item_type;
    auto literal_values = GetListLiteralValues(init);
    for (auto& imm_v : literal_values) {
      auto* imm_v_2 = RemoveMoveAndCast(imm_v.get());
      const auto& imm_type = RemoveReference(imm_v_2->checked_type());
      if (item_type != imm_type) {
        return Type(nullptr);
      }
    }
    return ListType(true, item_type, type_node->span);
  }
  if (auto* type_node = type.as<SetTypeNode>()) {
    const Type& item_type = type_node->item_type;
    auto literal_values = GetListLiteralValues(init);
    for (auto& imm_v : literal_values) {
      auto* imm_v_2 = RemoveMoveAndCast(imm_v.get());
      const auto& imm_type = RemoveReference(imm_v_2->checked_type());
      if (item_type != imm_type) {
        return Type(nullptr);
      }
    }
    return SetType(true, item_type, type_node->span);
  }
  if (auto* type_node = type.as<DictTypeNode>()) {
    const Type& key_type = type_node->key_type;
    const Type& value_type = type_node->value_type;
    auto literal_values = GetDictLiteralValues(init);
    for (auto& imm_kv : literal_values) {
      auto* imm_k = RemoveMoveAndCast(imm_kv.first.get());
      const auto& imm_key_type = RemoveReference(imm_k->checked_type());
      if (key_type != imm_key_type) {
        return Type(nullptr);
      }
      auto* imm_v = RemoveMoveAndCast(imm_kv.second.get());
      const auto& imm_value_type = RemoveReference(imm_v->checked_type());
      if (value_type != imm_value_type) {
        return Type(nullptr);
      }
    }
    return DictType(true, key_type, value_type, type_node->span);
  }
  return Type(nullptr);
}

static BaseExpr MutateListLiteralValues(const BaseExpr& init,
                                        const Type& type,
                                        const Type& item_type,
                                        const Span& span) {
  if (init->IsInstance<InitializerListNode>()) {
    Constructor ft_cons(type, "FTList", {item_type}, GlobalTypeVar(nullptr));
    return Call(type, ft_cons, {init}, span);
  }
  if (auto* call_node = init.as<CallNode>()) {
    {
      auto* constructor_op = call_node->op.as<ConstructorNode>();
      if (constructor_op && call_node->args.size() == 0) {
        Constructor ft_cons(type, "FTList", {item_type}, GlobalTypeVar(nullptr));
        return Call(type, ft_cons, {}, span);
      }
      if (constructor_op && call_node->args.size() == 1) {
        return MutateListLiteralValues(call_node->args[0], type, item_type, span);
      }
    }
    {
      if (call_node->op.same_as(builtin::list_fused_repeat_one()) && call_node->args.size() == 2) {
        return Call(type, builtin::ft_list_fused_repeat_one(), call_node->args, span);
      }
      if (call_node->op.same_as(builtin::list_fused_repeat_many()) && call_node->args.size() == 2) {
        return Call(type, builtin::ft_list_fused_repeat_many(), call_node->args, span);
      }
    }
  }
  return BaseExpr(nullptr);
}

static BaseExpr MutateSetLiteralValues(const BaseExpr& init,
                                       const Type& type,
                                       const Type& item_type,
                                       const Span& span) {
  if (init->IsInstance<InitializerListNode>()) {
    Constructor ft_cons(type, "FTSet", {item_type}, GlobalTypeVar(nullptr));
    return Call(type, ft_cons, {init}, span);
  }
  if (auto* call_node = init.as<CallNode>()) {
    auto* constructor_op = call_node->op.as<ConstructorNode>();
    if (constructor_op && call_node->args.size() == 0) {
      Constructor ft_cons(type, "FTSet", {item_type}, GlobalTypeVar(nullptr));
      return Call(type, ft_cons, {}, span);
    }
    if (constructor_op && call_node->args.size() == 1) {
      return MutateSetLiteralValues(call_node->args[0], type, item_type, span);
    }
  }
  return BaseExpr(nullptr);
}

static BaseExpr MutateDictLiteralValues(const BaseExpr& init,
                                        const Type& type,
                                        const Type& key_type,
                                        const Type& value_type,
                                        const Span& span) {
  if (init->IsInstance<InitializerDictNode>()) {
    Constructor ft_cons(type, "FTDict", {key_type, value_type}, GlobalTypeVar(nullptr));
    return Call(type, ft_cons, {init}, span);
  }
  if (auto* call_node = init.as<CallNode>()) {
    auto* constructor_op = call_node->op.as<ConstructorNode>();
    if (constructor_op && call_node->args.size() == 0) {
      Constructor ft_cons(type, "FTDict", {key_type, value_type}, GlobalTypeVar(nullptr));
      return Call(type, ft_cons, {}, span);
    }
    if (constructor_op && call_node->args.size() == 1) {
      return MutateDictLiteralValues(call_node->args[0], type, key_type, value_type, span);
    }
  }
  return BaseExpr(nullptr);
}

BaseExpr FullTypedOptimizerMutator::MutateLiteralValues(const BaseExpr& init, const Type& type) {
  if (const auto* node = type.as<ListTypeNode>()) {
    return MutateListLiteralValues(init, type, node->item_type, init->span);
  }
  if (const auto* node = type.as<SetTypeNode>()) {
    return MutateSetLiteralValues(init, type, node->item_type, init->span);
  }
  if (const auto* node = type.as<DictTypeNode>()) {
    return MutateDictLiteralValues(init, type, node->key_type, node->value_type, init->span);
  }
  return BaseExpr(nullptr);
}

FullTypedOptimizerMutator::FullTypedOptimizerMutator()
    : StmtExprMutator(),
      ops_mapping_{
          // list
          {builtin::list___len__().get(), builtin::ft_list___len__().get()},
          {builtin::list_reserve().get(), builtin::ft_list_reserve().get()},
          {builtin::list___getitem__().get(), builtin::ft_list___getitem__().get()},
          {builtin::list_capacity().get(), builtin::ft_list_capacity().get()},
          {builtin::list_pop().get(), builtin::ft_list_pop().get()},
          {builtin::list_clear().get(), builtin::ft_list_clear().get()},
          {builtin::list_reverse().get(), builtin::ft_list_reverse().get()},
          {builtin::list_sort_no_key().get(), builtin::ft_list_sort_no_key().get()},
          {builtin::list_sort().get(), builtin::ft_list_sort().get()},
          {builtin::list___contains__().get(), builtin::ft_list___contains__().get()},
          {builtin::list___setitem__().get(), builtin::ft_list___setitem__().get()},
          {builtin::list_append().get(), builtin::ft_list_append().get()},
          {builtin::list_index().get(), builtin::ft_list_index().get()},
          {builtin::list_insert().get(), builtin::ft_list_insert().get()},
          {builtin::list_remove().get(), builtin::ft_list_remove().get()},
          {builtin::list_count().get(), builtin::ft_list_count().get()},

          // TODO: set
          // TODO: dict
      } {
}

MATXSCRIPT_REGISTER_GLOBAL("ir.FullTypedOptimizer_GetMoveVarAndLineno")
    .set_body_typed([](BaseFunc f) {
      FullTypedOptimizerAnalysis analysis;
      auto result = analysis.run(f);
      std::vector<runtime::Tuple> info;
      for (auto& var_and_ty : result) {
        if (var_and_ty.first->span.defined()) {
          info.emplace_back(runtime::Tuple::dynamic(var_and_ty.first->name_hint(),
                                                    var_and_ty.first->span->lineno));
        } else {
          info.emplace_back(runtime::Tuple::dynamic(var_and_ty.first->name_hint(), -1));
        }
      }
      return runtime::Tuple(info.begin(), info.end());
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.FullTypedOptimizerMutator").set_body_typed([](BaseFunc f) {
  FullTypedOptimizerMutator optimizer;
  return runtime::RTValue(optimizer.run(f));
});

}  // namespace ir
}  // namespace matxscript
