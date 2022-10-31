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

#include <memory>
#include <random>
#include <string>
#include <vector>

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/bytes_hash.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class Graph;
class OpKernel;
struct Node;
struct NodeEntry;
class Symbol;
using NodeEntryPtr = std::shared_ptr<NodeEntry>;
using NodePtr = std::shared_ptr<Node>;

struct NodeEntry {
  NodePtr node = nullptr;
  uint32_t index = 0;
  String key;
  bool exported = false;
  runtime::RTValue data;

  NodeEntry() = default;
  NodeEntry(NodePtr node, uint32_t index, String key) {
    init(std::move(node), index, std::move(key));
  }

  void init(NodePtr n, uint32_t i, String k) {
    this->node = std::move(n);
    this->index = i;
    this->key = std::move(k);
  }

  NodeEntry(NodePtr node, uint32_t index, uint64_t node_sig) {
    init(std::move(node), index, node_sig);
  }

  void init(NodePtr n, uint32_t i, uint64_t n_sig) {
    this->node = std::move(n);
    this->index = i;
    char buf[256];
    int len = snprintf(buf, 256, "%llu_%u", n_sig, i);
    this->key.assign(buf, len);
  }

  bool operator==(const NodeEntry& o) const {
    return node == o.node && index == o.index;
  }

  String Name();
};

// Symbol is only used for save NodeEntryPtr
class Symbol {
 public:
  Symbol() : entry_(nullptr), all_entries_() {
  }
  explicit Symbol(const std::vector<NodeEntryPtr>& all_outputs, int64_t output_idx)
      : entry_(all_outputs[output_idx]), all_entries_(all_outputs) {
  }
  ~Symbol() = default;

 public:
  bool operator==(const Symbol& o) const {
    return entry_ == o.entry_;
  }
  bool operator!=(const Symbol& o) const {
    return entry_ != o.entry_;
  }
  const NodeEntryPtr& GetEntry() const {
    return entry_;
  }
  const std::vector<NodeEntryPtr>& GetAllEntries() const {
    return all_entries_;
  }

 private:
  NodeEntryPtr entry_ = nullptr;
  std::vector<NodeEntryPtr> all_entries_;
  friend class Graph;
  friend class TXSession;
};

struct NodeOutput {
  NodeEntry* source = nullptr;
  std::weak_ptr<NodeEntry> weak_ref;
};

inline std::vector<NodeOutput> MakeNodeOutputs(const std::vector<NodeEntryPtr>& entrys) {
  std::vector<NodeOutput> nos;
  nos.reserve(entrys.size());
  for (auto& ne : entrys) {
    NodeOutput o;
    o.source = ne.get();
    o.weak_ref = ne;
    nos.push_back(std::move(o));
  }
  return std::move(nos);
}

struct Node {
  OpKernelPtr op = nullptr;
  std::vector<NodeEntryPtr> inputs;
  std::vector<NodeOutput> outputs;
  std::vector<NodeEntryPtr> holder;
  String name;

  Node() {
    op = nullptr;
    name.reserve(128);
  }

  bool IsVariable() const;

  Dict ToDict() const;
  static NodePtr FromDict(const Dict& config, Graph* g);

  static NodePtr Create() {
    return std::make_shared<Node>();
  }
};

}  // namespace runtime
}  // namespace matxscript
