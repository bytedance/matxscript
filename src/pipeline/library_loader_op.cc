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
#include <matxscript/pipeline/library_loader_op.h>
#include "matxscript/runtime/container/string.h"
#include "matxscript/runtime/container/string_view.h"
#include "matxscript/runtime/cxxabi_helper.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace matxscript {
namespace runtime {

namespace {
class DSOLibrary {
 public:
  ~DSOLibrary() {
    if (lib_handle_)
      Unload();
  }
  DSOLibrary(const std::string& name) {
    Load(name);
  }

  void* GetSymbol(const char* name) {
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
  void Load(const std::string& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
    CHECK(lib_handle_ != nullptr) << "Failed to load dynamic shared library " << name;
  }

  void Unload() {
    FreeLibrary(lib_handle_);
    lib_handle_ = nullptr;
  }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const std::string& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
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
}  // namespace

void LibraryLoaderOp::load_dl_paths(const List& dl_paths) {
  if (resource_path_.empty()) {
    for (auto& path : dl_paths) {
      auto view = path.As<string_view>();
      lib_holder_.push_back(std::make_shared<DSOLibrary>(std::string(view.data(), view.size())));
    }
  } else {
    for (auto& path : dl_paths) {
      auto abs_path = resource_path_ + "/" + path.As<string_view>();
      auto view = abs_path.operator string_view();
      lib_holder_.push_back(std::make_shared<DSOLibrary>(std::string(view.data(), view.size())));
    }
  }
}

void LibraryLoaderOp::Init() {
  abi0_dl_paths_ = GetAttr<List>("abi0_dl_paths");
  abi1_dl_paths_ = GetAttr<List>("abi1_dl_paths");
  if (MATXSCRIPT_FLAGS_GLIBCXX_USE_CXX11_ABI) {
    load_dl_paths(abi1_dl_paths_);
  } else {
    load_dl_paths(abi0_dl_paths_);
  }
}

int LibraryLoaderOp::Bundle(string_view folder) {
  List new_abi0_paths, new_abi1_paths;
  for (auto& path : abi0_dl_paths_) {
    new_abi0_paths.append(BundlePath(path.As<string_view>(), folder));
  }
  for (auto& path : abi1_dl_paths_) {
    new_abi1_paths.append(BundlePath(path.As<string_view>(), folder));
  }
  SetAttr("abi0_dl_paths", std::move(new_abi0_paths));
  SetAttr("abi1_dl_paths", std::move(new_abi1_paths));
  return 0;
}

RTValue LibraryLoaderOp::Process(PyArgs inputs) const {
  MXCHECK(inputs.size() == 1) << "[LibcutOp] need 1 args, but receive: " << inputs.size();
  return inputs[0].As<RTValue>();
}

MATX_REGISTER_NATIVE_OP(LibraryLoaderOp);

}  // namespace runtime
}  // namespace matxscript