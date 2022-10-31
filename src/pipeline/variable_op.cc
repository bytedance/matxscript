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
#include <matxscript/pipeline/variable_op.h>

#include <matxscript/pipeline/node.h>

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(VariableOp);

std::unique_ptr<Symbol> VariableOp::make_symbol(OpKernelPtr op,
                                                String name,
                                                runtime::RTValue data) {
  NodePtr node = Node::Create();
  node->op = op;
  NodeEntryPtr entry = std::make_shared<NodeEntry>();
  entry->node = node;
  entry->key = name;
  entry->index = 0;
  entry->data = std::move(data);
  node->name = std::move(name);
  std::vector<NodeEntryPtr> nes{entry};
  node->outputs = MakeNodeOutputs(nes);
  return std::make_unique<Symbol>(nes, 0);
}

}  // namespace runtime
}  // namespace matxscript
