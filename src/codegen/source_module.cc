// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the source module is inspired by TVM.
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
 * \file source_module.cc
 * \brief Source code module, only for viewing
 */
#include "codegen_source_base.h"

#include <matxscript/runtime/container/ndarray.h>
#include <matxscript/runtime/file_util.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace codegen {

using namespace runtime;
using namespace ir;

// Simulator function
class SourceModuleNode : public runtime::ModuleNode {
 public:
  SourceModuleNode(std::string code, std::string fmt) : code_(code), fmt_(fmt) {
  }
  const char* type_key() const {
    return "source";
  }

  NativeFunction GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    MXLOG(FATAL) << "Source module cannot execute, to get executable module"
                 << " build MATXScript with \'" << fmt_ << "\' runtime support";
    return NativeFunction();
  }

  String GetSource(const String& format) final {
    return code_;
  }

 protected:
  String code_;
  String fmt_;
};

runtime::Module SourceModuleCreate(String code, String fmt) {
  auto n = make_object<SourceModuleNode>(code, fmt);
  return runtime::Module(n);
}

// Simulator function
class CSourceModuleNode : public runtime::ModuleNode {
 public:
  CSourceModuleNode(const String& code,
                    const String& fmt,
                    const String& symbol,
                    const Array<StringRef>& const_vars)
      : code_(code), fmt_(fmt), symbol_(symbol), const_vars_(const_vars) {
  }
  const char* type_key() const {
    return "c";
  }

  NativeFunction GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_symbol") {
      return NativeFunction([sptr_to_self, this](PyArgs args) -> RTValue { return this->symbol_; });
    } else if (name == "get_const_vars") {
      return NativeFunction(
          [sptr_to_self, this](PyArgs args) -> RTValue { return this->const_vars_; });
    } else {
      return NativeFunction(nullptr);
    }
  }

  String GetSource(const String& format) final {
    return code_;
  }

  void SaveToFile(const String& file_name, const String& format) final {
    String fmt = FileUtil::GetFileFormat(file_name, format);
    String meta_file = FileUtil::GetMetaFilePath(file_name);
    if (fmt == "cc") {
      MXCHECK_NE(code_.length(), 0);
      FileUtil::SaveBinaryToFile(file_name, code_);
    } else {
      MXCHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    }
  }

 protected:
  String code_;
  String fmt_;
  String symbol_;
  Array<StringRef> const_vars_;
};

runtime::Module CSourceModuleCreate(const String& code,
                                    const String& fmt,
                                    const String& symbol,
                                    const Array<StringRef>& const_vars) {
  auto n = make_object<CSourceModuleNode>(code, fmt, symbol, const_vars);
  return runtime::Module(n);
}

MATXSCRIPT_REGISTER_GLOBAL("runtime.SourceModuleCreate").set_body_typed(SourceModuleCreate);

MATXSCRIPT_REGISTER_GLOBAL("runtime.CSourceModuleCreate")
    .set_body_typed([](String code, String fmt, String symbol, Array<StringRef> const_vars) {
      return CSourceModuleCreate(code, fmt, symbol, const_vars);
    });

}  // namespace codegen
}  // namespace matxscript
