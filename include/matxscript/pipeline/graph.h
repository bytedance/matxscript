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
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <matxscript/pipeline/node.h>

namespace matxscript {
namespace runtime {

class TXSession;

class Graph {
 public:
  explicit Graph(const std::vector<NodePtr>& outputs);
  explicit Graph(const NodePtr& output_node);

  virtual ~Graph() = default;

 public:
  static std::shared_ptr<Graph> FromGenericList(TXSession* sess, List generic_graph);
  List ToGenericList() const;

 public:
  /**
   *
   * @return
   */
  const std::vector<NodePtr>& get_topo_nodes() const;

  /**
   *
   * @return
   */
  const std::vector<int>& get_input_nodes_index() const;

  /**
   *
   * @return
   */
  std::vector<NodePtr> get_input_nodes() const;

  /**
   *
   * @return
   */
  const std::vector<int>& get_output_nodes_index() const;

  /**
   *
   * @return
   */
  std::vector<NodePtr> get_output_nodes() const;

 private:
  explicit Graph() = default;
  void topo_sort(const std::vector<NodePtr>& outputs);
  void build_input_output(const std::vector<NodePtr>& outputs);
  static void dfs_visit(std::vector<NodePtr>& nodes,
                        std::unordered_set<NodePtr>& visited,
                        const NodePtr& node);

  void add_entry(NodeEntryPtr e) {
    name2entry_->emplace(e->key, e);
  }

 private:
  std::shared_ptr<std::vector<NodePtr>> nodes_;
  std::shared_ptr<std::vector<int>> inputs_;
  std::shared_ptr<std::vector<int>> outputs_;
  std::shared_ptr<std::unordered_map<std::string, NodeEntryPtr>> name2entry_;
  friend class Node;
};

}  // namespace runtime
}  // namespace matxscript
