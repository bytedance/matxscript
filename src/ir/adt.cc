// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Taken from https://github.com/apache/tvm/blob/v0.7/include/tvm/ir/adt.h
 * with fixes applied:
 * - add namespace matx::ir for fix conflict with tvm
 * - remove TypeData
 * - add ClassType
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
 * \file src/ir/adt.cc
 * \brief ADT type definitions.
 */
#include <matxscript/ir/adt.h>

#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace runtime;
using namespace ::matxscript::ir::printer;

Constructor::Constructor(Type ret_type,
                         StringRef name_hint,
                         Array<Type> inputs,
                         GlobalTypeVar belong_to) {
  ObjectPtr<ConstructorNode> n = make_object<ConstructorNode>();
  n->name_hint = std::move(name_hint);
  n->inputs = std::move(inputs);
  n->belong_to = std::move(belong_to);
  n->checked_type_ = std::move(ret_type);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ConstructorNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.Constructor")
    .set_body_typed(
        [](Type ret_type, StringRef name_hint, Array<Type> inputs, GlobalTypeVar belong_to) {
          return Constructor(
              std::move(ret_type), std::move(name_hint), std::move(inputs), std::move(belong_to));
        });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstructorNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ConstructorNode*>(ref.get());
      p->stream << "ConstructorNode(" << node->name_hint << ", " << node->inputs << ", "
                << node->belong_to << ")";
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<Constructor>("", [](Constructor node, ObjectPath p, IRDocsifier d) -> Doc {
      return IdDoc(node->name_hint);
    });

// ClassType
ClassType::ClassType(uint64_t py_type_id,
                     GlobalTypeVar header,
                     Type base,
                     Array<StringRef> var_names,
                     Array<Type> var_types,
                     Array<StringRef> func_names,
                     Array<StringRef> unbound_func_names,
                     Array<FuncType> func_types) {
  ObjectPtr<ClassTypeNode> n = make_object<ClassTypeNode>();
  n->py_type_id = py_type_id;
  n->header = std::move(header);
  n->var_names = std::move(var_names);
  n->var_types = std::move(var_types);
  n->func_names = std::move(func_names);
  n->unbound_func_names = std::move(unbound_func_names);
  n->func_types = std::move(func_types);
  n->base = std::move(base);
  if (n->base.defined()) {
    MXCHECK(n->base->IsInstance<ClassTypeNode>())
        << "class base type can only be class, but get " << n->base;
  }
  data_ = std::move(n);
}

Type ClassTypeNode::GetItem(const StringRef& name) const {
  auto itr_name = std::find(var_names.begin(), var_names.end(), name);
  auto itr_ub_func = std::find(unbound_func_names.begin(), unbound_func_names.end(), name);
  auto itr_func = std::find(func_names.begin(), func_names.end(), name);
  if (itr_name != var_names.end()) {
    int idx = std::distance(var_names.begin(), itr_name);
    return var_types[idx];
  } else if (itr_ub_func != unbound_func_names.end()) {
    int idx = std::distance(unbound_func_names.begin(), itr_ub_func);
    return func_types[idx];
  } else if (itr_func != func_names.end()) {
    int idx = std::distance(func_names.begin(), itr_func);
    return func_types[idx];
  } else {
    // do nothing, will return None
    if (base.defined()) {
      auto base_node = base.as<ClassTypeNode>();
      MXCHECK(base_node) << "class base type can only be class, but get " << base;
      return base_node->GetItem(name);
    }
    return Type();
  }
}

Array<StringRef> ClassTypeNode::GetVarNamesLookupTable() const {
  if (base.defined()) {
    // base0 vars
    // base1 vars
    // ...
    // cur vars
    auto base_node = base.as<ClassTypeNode>();
    MXCHECK(base_node) << "class base type can only be class, but get " << base;
    auto all_var_names = base_node->GetVarNamesLookupTable();
    all_var_names.insert(all_var_names.end(), var_names.begin(), var_names.end());
    return all_var_names;
  }
  return var_names;
}

Array<Type> ClassTypeNode::GetVarTypesLookupTable() const {
  if (base.defined()) {
    // base0 var types
    // base1 var types
    // ...
    // cur var types
    auto base_node = base.as<ClassTypeNode>();
    MXCHECK(base_node) << "class base type can only be class, but get " << base;
    auto all_var_types = base_node->GetVarTypesLookupTable();
    all_var_types.insert(all_var_types.end(), var_types.begin(), var_types.end());
    return all_var_types;
  }
  return var_types;
}

bool IsBaseTypeOf(const Type& base, const Type& derived, bool allow_same) {
  if (auto b_node = base.as<RefTypeNode>()) {
    return IsBaseTypeOf(b_node->value, derived, allow_same);
  }
  if (auto d_node = derived.as<RefTypeNode>()) {
    return IsBaseTypeOf(base, d_node->value, allow_same);
  }
  auto b_node = base.as<ClassTypeNode>();
  auto d_node = derived.as<ClassTypeNode>();
  if (b_node == nullptr || d_node == nullptr) {
    return false;
  }
  if (allow_same && b_node == d_node) {
    return true;
  }
  std::function<bool(const ClassTypeNode*, const ClassTypeNode*)> fn;
  fn = [&](const ClassTypeNode* base, const ClassTypeNode* derived) -> bool {
    if (derived->base.defined()) {
      auto derived_base = derived->base.as<ClassTypeNode>();
      MXCHECK(derived_base != nullptr);
      if (derived_base == base) {
        return true;
      } else {
        return fn(base, derived_base);
      }
    }
    return false;
  };
  return fn(b_node, d_node);
}

const PrimVar& GetImplicitClassSessionVar() {
  // session_var_name must be same as the var defined in ILightUserData
  static PrimVar var("this->session_handle_2_71828182846_", runtime::DataType::Handle());
  return var;
}

MATXSCRIPT_REGISTER_GLOBAL("ir.GetImplicitClassSessionVar").set_body([](PyArgs args) -> RTValue {
  return GetImplicitClassSessionVar();
});

MATXSCRIPT_REGISTER_NODE_TYPE(ClassTypeNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassType")
    .set_body_typed([](uint64_t py_type_id,
                       GlobalTypeVar header,
                       Type base,
                       Array<StringRef> var_names,
                       Array<Type> var_types,
                       Array<StringRef> func_names,
                       Array<StringRef> unbound_func_names,
                       Array<FuncType> func_types) {
      return ClassType(py_type_id,
                       std::move(header),
                       std::move(base),
                       std::move(var_names),
                       std::move(var_types),
                       std::move(func_names),
                       std::move(unbound_func_names),
                       std::move(func_types));
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ClassTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ClassTypeNode*>(ref.get());
      p->stream << "ClassType(name: " << node->header->name_hint << ")";
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassType_GetItem").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[ir.ClassType_GetItem] Expect 2 arguments but get " << args.size();
  ClassType cls_ty = args[0].As<ClassType>();
  StringRef name = args[1].As<StringRef>();
  auto ty = cls_ty->GetItem(name);
  if (ty.defined()) {
    return ty;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassType_InplaceAppendFunc").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 4) << "[ir.ClassType_AppendFunc] Expect 4 arguments but get "
                             << args.size();
  ClassType cls_ty = args[0].As<ClassType>();
  StringRef func_name = args[1].As<StringRef>();
  StringRef unbound_func_name = args[2].As<StringRef>();
  FuncType func_ty = args[3].As<FuncType>();
  auto* node = const_cast<ClassTypeNode*>(cls_ty.get());
  node->func_names.push_back(func_name);
  node->unbound_func_names.push_back(unbound_func_name);
  node->func_types.push_back(func_ty);
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassType_InplaceAppendVar").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[ir.ClassType_InplaceAppendVar] Expect 3 arguments but get "
                             << args.size();
  ClassType cls_ty = args[0].As<ClassType>();
  StringRef var_name = args[1].As<StringRef>();
  Type var_ty = args[2].As<Type>();
  auto* node = const_cast<ClassTypeNode*>(cls_ty.get());
  node->var_names.push_back(var_name);
  node->var_types.push_back(var_ty);
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassType_RebuildTag").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1 || args.size() == 2)
      << "[ir.ClassType_RebuildTag] Expect 1 or 2 arguments but get " << args.size();
  ClassType cls_ty = args[0].As<ClassType>();
  auto* node = const_cast<ClassTypeNode*>(cls_ty.get());
  String seed = node->header->name_hint + "_names";
  if (node->base.defined()) {
    auto base_node = node->base.as<ClassTypeNode>();
    MXCHECK(base_node != nullptr) << "class base type can only be class, but get " << node->base;
    seed.append("_");
    seed.append(std::to_string(base_node->tag));
    seed.append("_");
    seed.append(base_node->header->name_hint);
  }
  for (auto& var_name : node->var_names) {
    seed.append("_");
    seed.append(var_name);
  }
  seed.append("_types");
  for (auto& var_type : node->var_types) {
    seed.append("_");
    seed.append(var_type->GetTypeKey());
  }
  seed.append("_fn_names");
  for (auto& fn_name : node->func_names) {
    seed.append("_");
    seed.append(fn_name);
  }
  size_t tag = BytesHash(seed.c_str(), seed.size());
  if (args.size() == 2) {
    uint64_t mask = args[1].As<uint64_t>();
    tag &= mask;
  }
  node->tag = tag;
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.ClassType_ClearMembers").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[ir.ClassType_ClearMembers] Expect 1 arguments but get "
                            << args.size();
  ClassType cls_ty = args[0].As<ClassType>();
  auto* node = const_cast<ClassTypeNode*>(cls_ty.get());
  node->ClearMembers();
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("ir.IsBaseTypeOf").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 3) << "[ir.ClassType_IsBaseOf] Expect 3 arguments but get " << args.size();
  return IsBaseTypeOf(args[0].As<Type>(), args[1].As<Type>(), args[2].As<bool>());
});

}  // namespace ir
}  // namespace matxscript
