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
 * \file  module.cc
 * \brief The global module in Relay.
 */
// clang-format off
#include <matxscript/ir/module.h>

#include <fstream>
#include <sstream>
#include <unordered_set>

#include <matxscript/ir/_base/structural_equal.h>
#include <matxscript/runtime/registry.h>
// NOTE: reverse dependency on relay.
// These dependencies do not happen at the interface-level,
// and are only used in minimum cases where they are clearly marked.
//
// Rationale: We calls into relay's analysis module to verify correctness.
#include <matxscript/ir/type_functor.h>
#include <matxscript/ir/analysis.h>
#include <matxscript/ir/expr_functor.h>
// clang-format on

namespace matxscript {
namespace ir {

using namespace runtime;

IRModule::IRModule(Map<GlobalVar, BaseFunc> functions,
                   Map<GlobalTypeVar, ClassType> type_definitions,
                   std::unordered_set<StringRef> import_set) {
  auto n = make_object<IRModuleNode>();
  n->functions = std::move(functions);
  n->type_definitions = std::move(type_definitions);
  n->global_type_var_map_ = {};
  n->global_var_map_ = {};
  n->import_set_ = std::move(import_set);

  for (const auto& kv : n->functions) {
    // set global var map
    MXCHECK(n->global_var_map_.count(kv.first->name_hint) == 0)
        << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }

  for (const auto& kv : n->type_definitions) {
    // set global typevar map
    MXCHECK(n->global_type_var_map_.count(kv.first->name_hint) == 0)
        << "Duplicate global type definition name " << kv.first->name_hint;
    n->global_type_var_map_.Set(kv.first->name_hint, kv.first);
  }
  data_ = std::move(n);
}

bool IRModuleNode::SEqualReduce(const IRModuleNode* other, SEqualReducer equal) const {
  if (functions.size() != other->functions.size())
    return false;
  for (const auto& kv : this->functions) {
    if (!other->ContainGlobalVar(kv.first->name_hint))
      return false;
    if (!equal(kv.second, other->Lookup(kv.first->name_hint)))
      return false;
  }
  if (type_definitions.size() != other->type_definitions.size())
    return false;
  for (const auto& kv : this->type_definitions) {
    if (!other->ContainGlobalTypeVar(kv.first->name_hint))
      return false;
    if (!equal(kv.second, other->LookupTypeDef(kv.first->name_hint)))
      return false;
  }
  return true;
}

void IRModuleNode::SHashReduce(SHashReducer hash_reduce) const {
  using KV = std::pair<StringRef, ObjectRef>;
  // hash the functions.
  std::vector<KV> temp;

  auto reduce_temp = [&]() {
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(), [](const KV& lhs, const KV& rhs) {
      return lhs.first < rhs.first;
    });

    hash_reduce(static_cast<uint64_t>(temp.size()));
    // hash the content
    for (size_t i = 0; i < temp.size(); ++i) {
      hash_reduce(temp[i].first);
      hash_reduce(temp[i].second);
    }
  };

  for (const auto& kv : this->functions) {
    temp.emplace_back(kv.first->name_hint, kv.second);
  }
  reduce_temp();

  temp.clear();
  for (const auto& kv : this->type_definitions) {
    temp.emplace_back(kv.first->name_hint, kv.second);
  }
  reduce_temp();
}

bool IRModuleNode::ContainGlobalVar(const StringRef& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
}

bool IRModuleNode::ContainGlobalTypeVar(const StringRef& name) const {
  return global_type_var_map_.find(name) != global_type_var_map_.end();
}

GlobalVar IRModuleNode::GetGlobalVar(const StringRef& name) const {
  auto it = global_var_map_.find(name);
  if (it == global_var_map_.end()) {
    std::ostringstream msg;
    msg << "ValueError: Cannot find global var \"" << name << "\" in the Module\n"
        << "candidates are: [";
    int counter = 0;
    for (auto kv : global_var_map_) {
      if (counter++ != 0) {
        msg << ", ";
      }
      msg << "\"" << kv.first << "\"";
    }
    msg << "]";
    MXLOG(FATAL) << msg.str();
  }
  return (*it).second;
}

Array<GlobalVar> IRModuleNode::GetGlobalVars() const {
  std::vector<GlobalVar> global_vars;
  for (const auto& pair : global_var_map_) {
    global_vars.push_back(pair.second);
  }
  return Array<GlobalVar>(global_vars);
}

GlobalTypeVar IRModuleNode::GetGlobalTypeVar(const StringRef& name) const {
  MXCHECK(global_type_var_map_.defined());
  auto it = global_type_var_map_.find(name);
  MXCHECK(it != global_type_var_map_.end())
      << "Cannot find global type var " << name << " in the Module";
  return (*it).second;
}

Array<GlobalTypeVar> IRModuleNode::GetGlobalTypeVars() const {
  std::vector<GlobalTypeVar> global_type_vars;
  for (const auto& pair : global_type_var_map_) {
    global_type_vars.push_back(pair.second);
  }
  return Array<GlobalTypeVar>(global_type_vars);
}

void WarnIfMalformed(const IRModule& mod, Function func) {
  //  func = Downcast<relay::Function>(relay::DeDup(func));
  //  // Type check the item before we add it to the module.
  //  auto fv = relay::FreeVars(func);
  //  auto ftv = relay::FreeTypeVars(func, mod);
  //  // TODO(@jroesch): refactor to use diagnostic context
  //  CHECK_EQ(fv.size(), 0) << "There are free variables: " << fv << std::endl;
  //  CHECK_EQ(ftv.size(), 0) << "There are free type variables: " << fv
  //                          << " in function: " << AsText(func, false);
}

void IRModuleNode::AddExportFunction(const StringRef& func_name) {
  const auto* BaseFuncWithAttr =
      ::matxscript::runtime::FunctionRegistry::Get("ir.BaseFuncWithAttr");
  runtime::Map<GlobalVar, BaseFunc> new_functions;
  for (auto kv : functions) {
    if (kv.first->name_hint == func_name) {
      BaseFunc func =
          (*BaseFuncWithAttr)({kv.second, String(attr::kExportSymbol), Bool(true)}).As<BaseFunc>();
      new_functions.Set(kv.first, func);
    } else {
      new_functions.Set(kv.first, kv.second);
    }
  }
  functions = new_functions;
}

void IRModuleNode::Add(const GlobalVar& var, const BaseFunc& f, bool update) {
  BaseFunc checked_func = f;
  if (auto* ptr = f.as<FunctionNode>()) {
    WarnIfMalformed(GetRef<IRModule>(this), GetRef<Function>(ptr));
  }

  AddUnchecked(var, checked_func);
}

void IRModuleNode::AddUnchecked(const GlobalVar& var, const BaseFunc& func) {
  this->functions.Set(var, func);

  auto it = global_var_map_.find(var->name_hint);
  if (it != global_var_map_.end()) {
    MXCHECK_EQ((*it).second, var);
  } else {
    MXCHECK(global_var_map_.count(var->name_hint) == 0)
        << "Duplicate global function name " << var->name_hint;
  }

  global_var_map_.Set(var->name_hint, var);
}

void IRModuleNode::AddTypeDef(const GlobalTypeVar& var, const ClassType& type, bool update) {
  AddTypeDefUnchecked(var, type, update);
}

void IRModuleNode::AddTypeDefUnchecked(const GlobalTypeVar& var,
                                       const ClassType& type,
                                       bool update) {
  this->type_definitions.Set(var, type);
  if (!update) {
    // set global type var map
    MXCHECK(global_type_var_map_.count(var->name_hint) == 0)
        << "Duplicate global type definition name " << var->name_hint;
  }
  global_type_var_map_.Set(var->name_hint, var);
}

void IRModuleNode::Update(const GlobalVar& var, const BaseFunc& func) {
  this->Add(var, func, true);
}

void IRModuleNode::UpdateTypeDef(const GlobalTypeVar& var, const ClassType& type) {
  this->AddTypeDef(var, type, true);
}

void IRModuleNode::Remove(const GlobalVar& var) {
  auto functions_node = this->functions.CopyOnWrite();
  functions_node->erase(var);
  auto gvar_node = global_var_map_.CopyOnWrite();
  gvar_node->erase(var->name_hint);
}

BaseFunc IRModuleNode::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  MXCHECK(it != functions.end()) << "There is no definition of " << var->name_hint;
  return (*it).second;
}

BaseFunc IRModuleNode::Lookup(const StringRef& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

ClassType IRModuleNode::LookupTypeDef(const GlobalTypeVar& var) const {
  auto it = type_definitions.find(var);
  MXCHECK(it != type_definitions.end()) << "There is no definition of " << var->name_hint;
  return (*it).second;
}

ClassType IRModuleNode::LookupTypeDef(const StringRef& name) const {
  GlobalTypeVar id = this->GetGlobalTypeVar(name);
  return this->LookupTypeDef(id);
}

void IRModuleNode::Update(const IRModule& mod) {
  for (auto pair : mod->type_definitions) {
    this->AddTypeDef(pair.first, pair.second, false);
  }
  for (auto pair : mod->functions) {
    this->Add(pair.first, pair.second);
  }
}

IRModule IRModule::FromExpr(const HLOExpr& expr,
                            const Map<GlobalVar, BaseFunc>& global_funcs,
                            const Map<GlobalTypeVar, ClassType>& type_definitions) {
  auto mod = IRModule(global_funcs, type_definitions);
  BaseFunc func;
  StringRef gv_name = "main";

  if (auto* func_node = expr.as<BaseFuncNode>()) {
    func = GetRef<BaseFunc>(func_node);
    if (auto opt = func->GetAttr<StringRef>(attr::kGlobalSymbol)) {
      gv_name = opt.value();
    }
  } else {
    MXCHECK(false) << "[FromExpr] only support BaseFunc";
    // func = Function(relay::FreeVars(expr), {}, expr, Type(), relay::FreeTypeVars(expr, mod), {});
  }
  auto main_gv = GlobalVar(gv_name);
  mod->Add(main_gv, func);
  return mod;
}

std::unordered_set<StringRef> IRModuleNode::Imports() const {
  return this->import_set_;
}

MATXSCRIPT_REGISTER_NODE_TYPE(IRModuleNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.IRModule")
    .set_body_typed([](Map<GlobalVar, BaseFunc> funcs, Map<GlobalTypeVar, ClassType> types) {
      return IRModule(std::move(funcs), std::move(types), std::unordered_set<StringRef>());
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_Add").set_body([](PyArgs args) -> RTValue {
  IRModule mod = args[0].As<IRModule>();
  GlobalVar var = args[1].As<GlobalVar>();
  ObjectRef val = args[2].As<ObjectRef>();
  bool update = args[3].As<bool>();
  MXCHECK(val->IsInstance<HLOExprNode>());

  if (val->IsInstance<BaseFuncNode>()) {
    mod->Add(var, Downcast<BaseFunc>(val), update);
  } else if (val->IsInstance<GlobalVarNode>()) {
    GlobalVar gv = Downcast<GlobalVar>(val);
    auto mod_copy = IRModule(make_object<IRModuleNode>(*mod.operator->()));
    //    mod_copy = relay::transform::EtaExpand(
    //        /* expand_constructor */ false,
    //        /* expand_global_var */ true)(mod_copy);
    auto func = mod_copy->Lookup(gv->name_hint);
    mod->Add(var, Downcast<Function>(func), update);
  } else {
    auto func = Function({}, {}, Downcast<Stmt>(val), Type(nullptr), {});
    mod->Add(var, func, update);
  }
  return mod;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_AddDef")
    .set_body_typed([](IRModule mod, const GlobalTypeVar& var, const ClassType& type, bool update) {
      return mod->AddTypeDef(var, type, update);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_GetGlobalVar")
    .set_body_typed([](IRModule mod, const StringRef& str) { return mod->GetGlobalVar(str); });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_GetGlobalVars").set_body_typed([](IRModule mod) {
  return mod->GetGlobalVars();
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_GetGlobalTypeVars").set_body_typed([](IRModule mod) {
  return mod->GetGlobalTypeVars();
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_ContainGlobalVar")
    .set_body_typed([](IRModule mod, const StringRef& name) {
      return mod->ContainGlobalVar(name);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_GetGlobalTypeVar")
    .set_body_typed([](IRModule mod, const StringRef& name) {
      return mod->GetGlobalTypeVar(name);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_Lookup").set_body_typed([](IRModule mod, GlobalVar var) {
  return mod->Lookup(var);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_Lookup_str").set_body_typed([](IRModule mod, StringRef var) {
  return mod->Lookup(var);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_LookupDef")
    .set_body_typed([](IRModule mod, GlobalTypeVar var) { return mod->LookupTypeDef(var); });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_LookupDef_str")
    .set_body_typed([](IRModule mod, StringRef var) { return mod->LookupTypeDef(var); });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_FromExpr")
    .set_body_typed([](HLOExpr e,
                       Map<GlobalVar, BaseFunc> funcs,
                       Map<GlobalTypeVar, ClassType> type_defs) {
      return IRModule::FromExpr(e, funcs, type_defs);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_Update").set_body_typed([](IRModule mod, IRModule from) {
  mod->Update(from);
});

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_UpdateFunction")
    .set_body_typed([](IRModule mod, GlobalVar gv, BaseFunc func) { mod->Update(gv, func); });

MATXSCRIPT_REGISTER_GLOBAL("ir.Module_AddExportFunction")
    .set_body_typed([](IRModule mod, StringRef export_func) {
      mod->AddExportFunction(export_func);
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IRModuleNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IRModuleNode*>(ref.get());
      p->stream << "IRModule(" << node->functions << ")";
    });

}  // namespace ir
}  // namespace matxscript
