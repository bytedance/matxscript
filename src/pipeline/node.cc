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
#include <matxscript/pipeline/node.h>

#include <algorithm>

#include <matxscript/pipeline/graph.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

String NodeEntry::Name() {
  if (node) {
    return node->name + "[" + std::to_string(index) + "]";
  } else {
    return std::to_string(index);
  }
}

bool Node::IsVariable() const {
  return op->ClassName() == "VariableOp";
}

NodePtr Node::FromDict(const Dict& config, Graph* g) {
  String name_trace = config.get_default("name_trace", "").As<String>();

  auto node = Node::Create();
  node->name = std::move(name_trace);
  node->op = nullptr;

  // parse inputs
  MXCHECK(config.contains("inputs") && config["inputs"].IsObjectRef<List>());
  auto& inputs = config["inputs"];
  for (const auto& input : inputs.AsObjectRef<List>()) {
    MXCHECK(input.IsString()) << "inputs[i] must be string";
    auto i_name = input.As<String>();
    node->inputs.push_back(std::make_shared<NodeEntry>(nullptr, 0, i_name));
    node->inputs.back()->key = i_name;
  }

  // parse outputs
  MXCHECK(config.contains("outputs") && config["outputs"].IsObjectRef<List>());
  auto& outputs = config["outputs"];
  for (const auto& output : outputs.AsObjectRef<List>()) {
    MXCHECK(output.IsString()) << "expect outputs[i] is bytes, but get " << output.type_name();
    auto o_name = output.As<String>();
    auto entry = std::make_shared<NodeEntry>(nullptr, 0, o_name);
    g->add_entry(entry);
    NodeOutput no;
    no.source = entry.get();
    no.weak_ref = entry;
    node->outputs.push_back(no);
    node->outputs.back().source->key = o_name;
  }

  // parse exported
  if (config.contains("exported")) {
    auto& exported = config["exported"];
    MXCHECK(exported.IsObjectRef<List>()) << "exported must be array type";
    auto exported_list = exported.AsObjectRef<List>();
    for (auto itr = exported_list.begin(); itr != exported_list.end(); ++itr) {
      MXCHECK(itr->IsString()) << "exported[i] must be string type";
      String tmp = itr->As<String>();
      for (auto& entry : node->outputs) {
        if (entry.source->key == tmp) {
          entry.source->exported = true;
        }
      }
    }
  }

  return node;
}

Dict Node::ToDict() const {
  Dict generic_node;

  auto op_cls = op->ClassName();
  auto op_name = op->GetName();
  generic_node["op_cls"] = String(op_cls);
  generic_node["op_name"] = String(op_name);
  generic_node["name_trace"] = name;

  // op inputs
  List node_inputs;
  for (auto& input : inputs) {
    node_inputs.push_back(input->key);
  }
  generic_node["inputs"] = std::move(node_inputs);

  // op outputs
  List node_outputs;
  for (auto& output : outputs) {
    node_outputs.push_back(output.source->key);
  }
  generic_node["outputs"] = std::move(node_outputs);

  // op exported
  List node_exported;
  for (auto& output : outputs) {
    if (output.source->exported) {
      node_exported.push_back(output.source->key);
    }
  }
  if (!node_exported.empty()) {
    generic_node["exported"] = std::move(node_exported);
  }

  return generic_node;
}

}  // namespace runtime
}  // namespace matxscript
