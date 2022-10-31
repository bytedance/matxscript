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
#include <matxscript/pipeline/symbolic_executor.h>

#include <matxscript/pipeline/global_unique_index.h>

namespace matxscript {
namespace runtime {

std::vector<std::unique_ptr<Symbol>> SymbolicExecutor::Compose(
    OpKernelPtr op,
    const std::vector<const Symbol*>& args,
    const ska::flat_hash_map<String, const Symbol*>& kwargs,
    int num_output) {
  MXCHECK(kwargs.empty()) << "named args is not unsupported";
  // dynamic graphs generate symbols on demand,
  // so there is no need to repeat checks in the symbol executor

  // make a new node
  NodePtr node = Node::Create();
  node->op = op;
  node->name.append(op->name_).append("_node_");
  String input_node_names;
  for (size_t i = 0; i < args.size(); ++i) {
    auto& sym = args[i];
    node->inputs.push_back(sym->GetEntry());
    for (auto& o : sym->GetAllEntries()) {
      if (sym->GetEntry() != o) {
        node->holder.push_back(o);
      }
    }
    if (i + 1 == args.size()) {
      input_node_names.append(sym->GetEntry()->Name());
    } else {
      input_node_names.append(sym->GetEntry()->Name()).push_back('+');
    }
  }
  auto input_node_hash =
      std::to_string(BytesHash(input_node_names.data(), input_node_names.size()));
  node->name.append(input_node_hash.data(), input_node_hash.size());
  uint64_t node_key = GlobalUniqueIndex::instance()->gen_uniq_signature(node->name);

  std::vector<NodeEntryPtr> outputs;
  for (size_t i = 0; i < num_output; ++i) {
    auto entry_o_i = std::make_shared<NodeEntry>();
    entry_o_i->init(node, i, node_key);
    outputs.push_back(std::move(entry_o_i));
  }
  node->outputs = MakeNodeOutputs(outputs);
  std::vector<std::unique_ptr<Symbol>> symbolic_outputs;
  for (size_t i = 0; i < num_output; ++i) {
    symbolic_outputs.push_back(std::make_unique<Symbol>(outputs, i));
  }
  return symbolic_outputs;
}

}  // namespace runtime
}  // namespace matxscript
