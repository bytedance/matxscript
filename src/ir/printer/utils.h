// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from TVM.
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
#pragma once

#include <unordered_set>
#include <utility>
#include <vector>

#include "ir_docsifier.h"

#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/runtime/str_escape.h>

namespace matxscript {
namespace ir {
namespace printer {

using runtime::Downcast;
using runtime::GetRef;

extern StringRef DocToPythonScript(Doc doc, const PrinterConfig& cfg);

inline StringRef GenerateUniqueName(StringRef name_hint,
                                    const std::unordered_set<StringRef>& defined_names) {
  for (char& c : name_hint) {
    if (c != '_' && !std::isalnum(c)) {
      c = '_';
    }
  }
  StringRef name = name_hint;
  for (int i = 1; defined_names.count(name) > 0; ++i) {
    name = name_hint + "_" + std::to_string(i);
  }
  return name;
}

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
