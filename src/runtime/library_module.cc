// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
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
 * \file module_util.cc
 * \brief Utilities for module.
 */
#include "library_module.h"

#include <utility>
#include <vector>

#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/func_registry_names_io.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/module.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

NativeFunction WrapPackedFunc(MATXScriptBackendPackedCFunc faddr,
                              const ObjectPtr<Object>& sptr_to_self,
                              bool capture_session_handle) {
  if (capture_session_handle) {
    return NativeFunction([faddr, sptr_to_self](PyArgs args) -> RTValue {
      MXCHECK(args.size() > 0) << "closures requires at least one handle parameter";
      auto* handle = args[args.size() - 1].As<void*>();
      args = PyArgs(args.begin(), args.size() - 1);
      MATXScriptAny ret_value;
      std::vector<MATXScriptAny> c_args;
      c_args.reserve(args.size());
      for (auto& val : args) {
        c_args.push_back(val.value());
      }
      int ret = (*faddr)(c_args.data(), args.size(), &ret_value, handle);
      MXCHECK_EQ(ret, 0) << MATXScriptAPIGetLastError();
      return RTValue::MoveFromCHost(&ret_value);
    });
  } else {
    return NativeFunction([faddr, sptr_to_self](PyArgs args) -> RTValue {
      MATXScriptAny ret_value;
      std::vector<MATXScriptAny> c_args;
      c_args.reserve(args.size());
      for (auto& val : args) {
        c_args.push_back(val.value());
      }
      int ret = (*faddr)(c_args.data(), args.size(), &ret_value, nullptr);
      MXCHECK_EQ(ret, 0) << MATXScriptAPIGetLastError();
      return RTValue::MoveFromCHost(&ret_value);
    });
  }
}

// Library module that exposes symbols from a library.
class LibraryModuleNode final : public ModuleNode {
 public:
  explicit LibraryModuleNode(ObjectPtr<Library> lib) : lib_(std::move(lib)) {
    auto* func_reg = reinterpret_cast<MATXScriptFuncRegistry*>(
        lib_->GetSymbol(runtime::symbol::library_func_registry));
    MXCHECK(func_reg != nullptr) << "Symbol " << runtime::symbol::library_func_registry
                                 << " is not presented";
    auto func_reg_names = ReadFuncRegistryNames(func_reg->names);
    for (size_t i = 0; i < func_reg_names.size(); ++i) {
      func_regs_.emplace(func_reg_names[i], func_reg->funcs[i]);
    }
    auto* closures_names_ptr =
        reinterpret_cast<const char**>(lib_->GetSymbol(runtime::symbol::library_closures_names));
    MXCHECK(closures_names_ptr != nullptr)
        << "Symbol " << runtime::symbol::library_closures_names << " is not presented";
    auto closures_names = ReadFuncRegistryNames(*closures_names_ptr);
    for (size_t i = 0; i < closures_names.size(); ++i) {
      closures_names_.emplace(closures_names[i]);
    }
  }

  const char* type_key() const final {
    return "library";
  }

  NativeFunction GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    bool capture_session_handle = closures_names_.find(name) != closures_names_.end();
    MATXScriptBackendPackedCFunc faddr;
    auto func_reg_itr = func_regs_.find(name);
    if (func_reg_itr != func_regs_.end()) {
      return WrapPackedFunc(func_reg_itr->second, sptr_to_self, capture_session_handle);
    }
    if (name == runtime::symbol::library_func_registry) {
      auto* func_reg = reinterpret_cast<MATXScriptFuncRegistry*>(
          lib_->GetSymbol(runtime::symbol::library_func_registry));
      TypedNativeFunction<void*()> pf([func_reg, sptr_to_self]() { return func_reg; });
      return pf.packed();
    } else {
      faddr = reinterpret_cast<MATXScriptBackendPackedCFunc>(lib_->GetSymbol(name.c_str()));
    }
    if (faddr == nullptr)
      return NativeFunction();
    return WrapPackedFunc(faddr, sptr_to_self, capture_session_handle);
  }

 private:
  ObjectPtr<Library> lib_;
  ska::flat_hash_map<String, MATXScriptBackendPackedCFunc> func_regs_;
  ska::flat_hash_set<String> closures_names_;
};

Module CreateModuleFromLibrary(ObjectPtr<Library> lib) {
  auto n = make_object<LibraryModuleNode>(lib);
  Module root_mod = Module(n);

  // allow lookup of symbol from root (so all symbols are visible).
  if (auto* ctx_addr =
          reinterpret_cast<void**>(lib->GetSymbol(runtime::symbol::library_module_ctx))) {
    *ctx_addr = root_mod.operator->();
  }

  return root_mod;
}
}  // namespace runtime
}  // namespace matxscript
