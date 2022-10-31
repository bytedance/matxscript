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
 * \file dso_library.cc
 * \brief Create library module to load from dynamic shared library.
 */
#include "library_module.h"

#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/module.h>
#include <matxscript/runtime/registry.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace matxscript {
namespace runtime {

// Dynamic shared library.
// This is the default module TVM used for host-side AOT
class DSOLibrary final : public Library {
 public:
  ~DSOLibrary() {
    if (lib_handle_)
      Unload();
  }
  void Init(const String& name) {
    Load(name);
  }

  void* GetSymbol(const char* name) final {
    return GetSymbol_(name);
  }

 private:
  // Platform dependent handling.
#if defined(_WIN32)
  // library handle
  HMODULE lib_handle_{nullptr};

  void* GetSymbol_(const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(lib_handle_, (LPCSTR)name));  // NOLINT(*)
  }

  // Load the library
  void Load(const String& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
    MXCHECK(lib_handle_ != nullptr) << "Failed to load dynamic shared library " << name;
  }

  void Unload() {
    FreeLibrary(lib_handle_);
    lib_handle_ = nullptr;
  }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const String& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    MXCHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name << " " << dlerror();
  }

  void* GetSymbol_(const char* name) {
    return dlsym(lib_handle_, name);
  }

  void Unload() {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
#endif
};

MATXSCRIPT_REGISTER_GLOBAL("runtime.module.loadfile_so").set_body([](PyArgs args) -> RTValue {
  auto n = make_object<DSOLibrary>();
  n->Init(args[0].As<String>());
  return CreateModuleFromLibrary(n);
});

}  // namespace runtime
}  // namespace matxscript
