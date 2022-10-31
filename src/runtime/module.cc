// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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
 * \file module.cc
 * \brief MATX module system
 */
#include <matxscript/runtime/module.h>

#include <cstring>
#include <unordered_set>

#include <matxscript/runtime/file_util.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

void ModuleNode::Import(Module other) {
  // specially handle rpc
  if (!std::strcmp(this->type_key(), "rpc")) {
    static const NativeFunction* fimport_ = nullptr;
    if (fimport_ == nullptr) {
      fimport_ = runtime::FunctionRegistry::Get("rpc.ImportRemoteModule");
      MXCHECK(fimport_ != nullptr);
    }
    (*fimport_)({GetRef<Module>(this), other});
    return;
  }
  // cyclic detection.
  std::unordered_set<const ModuleNode*> visited{other.operator->()};
  std::vector<const ModuleNode*> stack{other.operator->()};
  while (!stack.empty()) {
    const ModuleNode* n = stack.back();
    stack.pop_back();
    for (const Module& m : n->imports_) {
      const ModuleNode* next = m.operator->();
      if (visited.count(next))
        continue;
      visited.insert(next);
      stack.push_back(next);
    }
  }
  MXCHECK(!visited.count(this)) << "Cyclic dependency detected during import";
  this->imports_.emplace_back(std::move(other));
}

NativeFunction ModuleNode::GetFunction(const String& name, bool query_imports) {
  ModuleNode* self = this;
  NativeFunction pf = self->GetFunction(name, GetObjectPtr<Object>(this));
  if (pf != nullptr)
    return pf;
  if (query_imports) {
    for (Module& m : self->imports_) {
      pf = m.operator->()->GetFunction(name, query_imports);
    }
  }
  return pf;
}

Module Module::LoadFromFile(const String& file_name, const String& format) {
  String fmt = FileUtil::GetFileFormat(file_name, format);
  MXCHECK(fmt.length() != 0) << "Cannot deduce format of file " << file_name;
  if (fmt == "dll" || fmt == "dylib" || fmt == "dso") {
    fmt = "so";
  }
  String load_f_name = "runtime.module.loadfile_" + fmt;
  const NativeFunction* f = FunctionRegistry::Get(load_f_name);
  MXCHECK(f != nullptr) << "Loader of " << format << "(" << load_f_name << ") is not presented.";
  Module m = (*f)({String(file_name), String(format)}).As<Module>();
  return m;
}

void ModuleNode::SaveToFile(const String& file_name, const String& format) {
  MXLOG(FATAL) << "Module[" << type_key() << "] does not support SaveToFile";
}

String ModuleNode::GetSource(const String& format) {
  MXLOG(FATAL) << "Module[" << type_key() << "] does not support GetSource";
  return "";
}

const NativeFunction* ModuleNode::GetFuncFromEnv(const String& name) {
  auto it = import_cache_.find(name);
  if (it != import_cache_.end())
    return it->second.get();
  NativeFunction pf;
  for (Module& m : this->imports_) {
    pf = m.GetFunction(name, true);
    if (pf != nullptr)
      break;
  }
  if (pf == nullptr) {
    const NativeFunction* f = FunctionRegistry::Get(name);
    MXCHECK(f != nullptr) << "Cannot find function " << name
                          << " in the imported modules or global registry";
    return f;
  } else {
    import_cache_.insert(std::make_pair(name, std::make_shared<NativeFunction>(pf)));
    return import_cache_.at(name).get();
  }
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(ModuleNode);
}  // namespace runtime
}  // namespace matxscript
