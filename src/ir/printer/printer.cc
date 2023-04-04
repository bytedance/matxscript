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
#include <matxscript/ir/printer/printer.h>

#include <matxscript/ir/_base/object_path.h>
#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/ir/printer/text_printer.h>

namespace matxscript {
namespace ir {

std::ostream& operator<<(std::ostream& os, const runtime::ObjectRef& n) {
  if (const auto* path_node = n.as<ObjectPathNode>()) {
    os << path_node->GetRepr();
  } else {
    static ir::printer::PrinterConfig config;
    os << ir::printer::IRTextPrinter::Print(n, config);
  }
  return os;
}

}  // namespace ir
}  // namespace matxscript
