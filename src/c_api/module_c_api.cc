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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/module.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("runtime.ModuleGetSource").set_body_typed([](Module mod, Unicode fmt) {
  return String(mod->GetSource(fmt.encode())).decode();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ModuleImportsSize").set_body_typed([](Module mod) {
  return static_cast<int64_t>(mod->imports().size());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ModuleGetImport").set_body_typed([](Module mod, int index) {
  return mod->imports().at(index);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ModuleGetTypeKey").set_body_typed([](Module mod) {
  return String(mod->type_key()).decode();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ModuleLoadFromFile")
    .set_body_typed([](Unicode name, Unicode fmt) {
      return Module::LoadFromFile(name.encode(), fmt.encode());
    });

MATXSCRIPT_REGISTER_GLOBAL("runtime.ModuleSaveToFile")
    .set_body_typed([](Module mod, Unicode name, Unicode fmt) {
      mod->SaveToFile(name.encode(), fmt.encode());
    });

MATXSCRIPT_REGISTER_GLOBAL("runtime.GetCodegenModulePtrName").set_body_typed([]() {
  return String(symbol::library_module_ctx).decode();
});

}  // namespace runtime
}  // namespace matxscript
