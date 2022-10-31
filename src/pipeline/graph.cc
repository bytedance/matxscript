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
#include <matxscript/pipeline/graph.h>

#include <matxscript/pipeline/node.h>
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

Graph::Graph(const std::vector<NodePtr>& outputs) {
  topo_sort(outputs);
  build_input_output(outputs);
}

Graph::Graph(const NodePtr& output_node) {
  std::vector<NodePtr> outputs = {output_node};
  topo_sort(outputs);
  build_input_output(outputs);
}

void Graph::topo_sort(const std::vector<NodePtr>& outputs) {
  name2entry_ = std::make_shared<std::unordered_map<std::string, NodeEntryPtr>>();
  nodes_ = std::make_shared<std::vector<NodePtr>>();
  std::unordered_set<NodePtr> visited;
  for (auto& node : outputs) {
    dfs_visit(*nodes_, visited, node);
  }
}

void Graph::build_input_output(const std::vector<NodePtr>& outputs) {
  inputs_ = std::make_shared<std::vector<int>>();
  outputs_ = std::make_shared<std::vector<int>>();
  std::unordered_set<NodePtr> outputs_set(outputs.begin(), outputs.end());
  for (int i = 0; i < nodes_->size(); ++i) {
    auto& node = nodes_->at(i);
    if (node->IsVariable()) {
      inputs_->push_back(i);
    }
    if (outputs_set.find(node) != outputs_set.end()) {
      outputs_->push_back(i);
    }
  }
}

void Graph::dfs_visit(std::vector<NodePtr>& nodes,
                      std::unordered_set<NodePtr>& visited,
                      const NodePtr& node) {
  if (!node) {
    return;
  }
  for (auto& sub_node : node->inputs) {
    // dfs_visit(nodes, visited, sub_node);
    dfs_visit(nodes, visited, sub_node->node);
  }
  if (visited.find(node) == visited.end()) {
    nodes.emplace_back(node);
    visited.emplace(node);
  }
}

const std::vector<int>& Graph::get_input_nodes_index() const {
  return *inputs_;
}

std::vector<NodePtr> Graph::get_input_nodes() const {
  std::vector<NodePtr> result;
  for (auto& i : *inputs_) {
    result.emplace_back(nodes_->at(i));
  }
  return std::move(result);
}

const std::vector<int>& Graph::get_output_nodes_index() const {
  return *outputs_;
}

std::vector<NodePtr> Graph::get_output_nodes() const {
  std::vector<NodePtr> result;
  for (auto& i : *outputs_) {
    result.emplace_back(nodes_->at(i));
  }
  return std::move(result);
}

const std::vector<NodePtr>& Graph::get_topo_nodes() const {
  return *nodes_;
}

std::shared_ptr<Graph> Graph::FromGenericList(TXSession* sess, List generic_graph) {
  std::shared_ptr<Graph> graph;
  graph.reset(new Graph);
  graph->name2entry_ = std::make_shared<std::unordered_map<std::string, NodeEntryPtr>>();
  graph->nodes_ = std::make_shared<std::vector<NodePtr>>();
  // init ops and build graph
  std::unordered_map<std::string, NodePtr> key2node;
  std::vector<NodePtr> node_outputs;
  for (const auto& node_obj : generic_graph) {
    Dict node_config = node_obj.AsObjectRef<Dict>();
    String op_cls = node_config.get_item("op_cls").As<String>();
    String op_name = node_config.get_item("op_name").As<String>();
    NodePtr node = Node::FromDict(node_config, graph.get());
    auto op_ptr = sess->FindOp(op_cls, op_name);
    MXCHECK(op_ptr != nullptr) << "not found op: " << op_cls << ", name: " << op_name;
    node->op = op_ptr;
    graph->nodes_->emplace_back(node);
    for (size_t i = 0; i < node->outputs.size(); ++i) {
      auto& entry = node->outputs[i];
      entry.source->node = node;
      entry.source->index = i;
      key2node.emplace(entry.source->key, node);
      if (entry.source->exported) {
        node_outputs.push_back(node);
      }
    }
  }
  for (auto& node : *graph->nodes_) {
    for (auto& entry : node->inputs) {
      entry->node = key2node.at(entry->key);
      for (size_t i = 0; i < entry->node->outputs.size(); ++i) {
        if (entry->node->outputs[i].source->key == entry->key) {
          entry->index = i;
        }
      }
    }
  }
  MXCHECK(!node_outputs.empty()) << "compute graph has zero output nodes!!!";
  graph->build_input_output(node_outputs);
  return std::move(graph);
}

List Graph::ToGenericList() const {
  List generic_graph;
  generic_graph.reserve(nodes_->size());
  for (size_t i = 0; i < nodes_->size(); ++i) {
    auto& node = nodes_->at(i);
    generic_graph.push_back(node->ToDict());
  }
  return generic_graph;
}

}  // namespace runtime
}  // namespace matxscript
