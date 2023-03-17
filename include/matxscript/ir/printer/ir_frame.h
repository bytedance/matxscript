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

#include <matxscript/ir/printer/ir_docsifier.h>

namespace matxscript {
namespace ir {
namespace printer {

using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;

class IRFrameNode : public FrameNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
  }

  static constexpr const char* _type_key = "ir.printer.IRFrame";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IRFrameNode, FrameNode);
};

class IRFrame : public Frame {
 public:
  explicit IRFrame(const IRDocsifier& d) {
    ObjectPtr<IRFrameNode> n = runtime::make_object<IRFrameNode>();
    n->stmts.clear();
    n->d = d.get();
    data_ = std::move(n);
  }

  MATXSCRIPT_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRFrame, Frame, IRFrameNode);
};

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
