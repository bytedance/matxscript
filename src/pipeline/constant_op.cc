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
#include <matxscript/pipeline/constant_op.h>

#include <matxscript/pipeline/node.h>
#include <matxscript/pipeline/tx_session.h>

#include "userdata_mutator.h"

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(ConstantOp);

void ConstantOp::Init() {
  MXCHECK(HasAttr("data"));
  data = GetAttr<RTValue>("data");
  UserDataMutator::Mutate(&data, this);
}

std::unique_ptr<Symbol> ConstantOp::make_symbol(std::shared_ptr<ConstantOp> op) {
  NodePtr node = Node::Create();
  node->op = op;
  node->name = op->GetName();
  uint64_t node_key = GlobalUniqueIndex::instance()->gen_uniq_signature(node->name);
  std::vector<NodeEntryPtr> nes;
  nes.push_back(std::make_shared<NodeEntry>(node, 0, node_key));
  nes.back()->data = op->data;
  node->outputs = MakeNodeOutputs(nes);
  return std::make_unique<Symbol>(nes, 0);
}

RTValue ConstantOp::Process(PyArgs inputs) const {
  CheckArgs(inputs.size(), 0);
  return data;
}

}  // namespace runtime
}  // namespace matxscript
