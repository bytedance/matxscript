// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
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
/*!
 * \file src/node/structural_hash.cc
 */
#include <matxscript/ir/_base/structural_hash.h>

#include <algorithm>
#include <unordered_map>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace runtime;

// Define the dispatch function here since primary user is in this file.
void ReflectionVTable::SHashReduce(const Object* self, SHashReducer reducer) const {
  uint32_t tindex = self->type_index();
  if (tindex >= fshash_reduce_.size() || fshash_reduce_[tindex] == nullptr) {
    MXLOG(FATAL) << "TypeError: SHashReduce of " << self->GetTypeKey()
                 << " is not registered via TVM_REGISTER_NODE_TYPE";
  }
  fshash_reduce_[tindex](self, reducer);
}

// Hash handler that handles free vars
// by assigning a unique counter in the order of their occurrence.
//
// This algorithm depends on the determinism of the traversal of SHash function.
// In particular, when we traverse unordered_map, we should first sort
// the entries by keys(or hash of keys) before traversing.

class SHashHandlerDefault::Impl {
 public:
  explicit Impl(SHashHandlerDefault* parent) : parent_(parent) {
  }

  /*! \brief Pending reduce tasks. */
  struct Task {
    /*!
     * \brief The object operand to be hashed.
     *  If the object is nullptr, then the reduced hash is already set
     *  the correct value.
     */
    ObjectRef object;
    /*! \brief The partially reduce hash value.*/
    uint64_t reduced_hash;
    /*! \brief The expected location in the result stack. */
    uint64_t result_stack_index = std::numeric_limits<uint64_t>::max();
    /*! \brief Whether the children has been expanded via SEqualReduce */
    bool children_expanded{false};
    /*! \brief Whether the node is graph node. */
    bool graph_node_hash{false};
    /*! \brief whether to map the free variables. */
    bool map_free_vars;

    Task() = default;
    explicit Task(ObjectRef object, uint64_t reduced_hash, bool map_free_vars)
        : object(object), reduced_hash(reduced_hash), map_free_vars(map_free_vars) {
    }
  };

  void MarkGraphNode() {
    // need to push to pending tasks in this case
    MXCHECK(!allow_push_to_stack_ && !task_stack_.empty());
    task_stack_.back().graph_node_hash = true;
  }

  bool LookupHashedValue(const ObjectRef& key, uint64_t* hash_value) {
    auto it = hash_memo_.find(key);
    if (it != hash_memo_.end()) {
      hash_value[0] = it->second;
      return true;
    }
    return false;
  }

  void SHashReduceHashedValue(uint64_t hashed_value) {
    pending_tasks_.emplace_back(Task(ObjectRef(nullptr), hashed_value, false));
  }

  void SHashReduceFreeVar(const runtime::Object* var, bool map_free_vars) {
    MXCHECK(!hash_memo_.count(GetRef<ObjectRef>(var)));
    if (map_free_vars) {
      // use counter value.
      uint64_t value = std::hash<uint64_t>()(free_var_counter_++);
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), value, false));
    } else {
      // use pointer hash
      uint64_t value = std::hash<const runtime::Object*>()(var);
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), value, false));
    }
  }

  void SHashReduce(const ObjectRef& object, bool map_free_vars) {
    // Directly push the result
    // Note: it is still important to push the result to pending tasks
    // so that the reduction order of hash values stays the same.
    if (!object.defined()) {
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), 0, false));
      return;
    }
    auto it = hash_memo_.find(object);
    if (it != hash_memo_.end()) {
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), it->second, false));
    } else {
      // Push a pending task with initial value.
      pending_tasks_.emplace_back(Task(object, object->GetTypeKeyHash(), map_free_vars));
    }
  }

  uint64_t Hash(const ObjectRef& object, bool map_free_vars) {
    MXCHECK_EQ(task_stack_.size(), 0U);
    MXCHECK_EQ(pending_tasks_.size(), 0U);
    MXCHECK_EQ(result_stack_.size(), 0U);

    this->SHashReduce(object, map_free_vars);
    MXCHECK_EQ(pending_tasks_.size(), 1U);
    MXCHECK(allow_push_to_stack_);
    task_stack_.emplace_back(std::move(pending_tasks_.back()));
    pending_tasks_.clear();

    this->RunTasks();

    MXCHECK_EQ(result_stack_.size(), 1U);
    uint64_t ret = result_stack_.back();
    result_stack_.pop_back();
    return ret;
  }

  void DispatchSHash(const ObjectRef& object, bool map_free_vars) {
    MXCHECK(object.defined());
    vtable_->SHashReduce(object.get(), SHashReducer(parent_, map_free_vars));
  }

 protected:
  /*!
   * \brief Pop the top entry of the task stack and push the hash into the result stack.
   */
  void PopTaskStack() {
    const auto& entry = task_stack_.back();
    result_stack_.push_back(entry.reduced_hash);
    task_stack_.pop_back();
  }
  /*!
   * \brief Compute the reduced hash value for the task.
   * \param task The indicated task.
   */
  uint64_t ReduceHash(const Task& task) {
    uint64_t stack_begin = task.result_stack_index;
    MXCHECK_LE(stack_begin, result_stack_.size());

    // combine in the reverse order of the stack.
    uint64_t reduced_hash = task.reduced_hash;
    for (uint32_t i = result_stack_.size(); i != stack_begin; --i) {
      reduced_hash = HashCombine(reduced_hash, result_stack_[i - 1]);
    }
    result_stack_.resize(stack_begin);
    return reduced_hash;
  }
  // run the tasks.
  void RunTasks() {
    while (task_stack_.size() != 0) {
      // Caution: entry becomes invalid when the stack changes
      auto& entry = task_stack_.back();
      if (entry.children_expanded) {
        // reduce hash
        entry.reduced_hash = ReduceHash(entry);
        // When all the children has expanded and visited.
        // entry.reduced_hash contains the reduced hash result.
        auto it = hash_memo_.find(entry.object);
        if (it != hash_memo_.end()) {
          // use the pre-computed hash for the object.
          entry.reduced_hash = it->second;
        } else {
          // Append the graph node counter to the hash
          // so that we can distinguish DAG from trees.
          if (entry.graph_node_hash) {
            entry.reduced_hash =
                HashCombine(entry.reduced_hash, std::hash<uint64_t>()(graph_node_counter_++));
          }
          hash_memo_[entry.object] = entry.reduced_hash;
        }
        // send value to parent.
        this->PopTaskStack();
      } else if (!entry.object.defined()) {
        // Directly send value to parent
        this->PopTaskStack();
      } else {
        // check if there are already hash for object.
        auto it = hash_memo_.find(entry.object);
        if (it != hash_memo_.end()) {
          entry.reduced_hash = it->second;
          this->PopTaskStack();
        } else {
          // NOTE: important to modify entry before visit.
          // as entry becomes invalid after we change the stack.
          entry.children_expanded = true;
          entry.result_stack_index = result_stack_.size();

          MXCHECK_EQ(pending_tasks_.size(), 0U);
          allow_push_to_stack_ = false;
          // dispatch hash, reduce to the current slot.
          parent_->DispatchSHash(entry.object, entry.map_free_vars);
          allow_push_to_stack_ = true;
          // Move pending tasks to the stack until the marked point.
          while (pending_tasks_.size() != 0) {
            task_stack_.emplace_back(std::move(pending_tasks_.back()));
            pending_tasks_.pop_back();
          }
        }
      }
    }
  }

  /*!
   * \brief Combine two hash values into a single one.
   * \param key The left operand.
   * \param value The right operand.
   * \return the combined result.
   */
  size_t HashCombine(size_t key, size_t value) {
    // XXX: do not use std::hash in this function. This hash must be stable
    // across different platforms and std::hash is implementation dependent.
    return key ^ (uint64_t(value) + 0x9e3779b9 + (key << 6) + (key >> 2));
  }

 private:
  // The owner of this impl
  SHashHandlerDefault* parent_;
  // free var counter.
  uint32_t free_var_counter_{0};
  // graph node counter.
  uint32_t graph_node_counter_{0};
  // record current stack top
  bool allow_push_to_stack_{true};
  // list of pending tasks to be pushed to the stack.
  std::vector<Task> pending_tasks_;
  // Internal task stack to executed the task
  std::vector<Task> task_stack_;
  // Internal stack to store the result popped from the task stack.
  std::vector<uint64_t> result_stack_;
  // reflection vtable
  ReflectionVTable* vtable_ = ReflectionVTable::Global();
  // map from lhs to rhs
  std::unordered_map<ObjectRef, uint64_t, ObjectPtrHash, ObjectPtrEqual> hash_memo_;
};

SHashHandlerDefault::SHashHandlerDefault() {
  impl = new Impl(this);
}
SHashHandlerDefault::~SHashHandlerDefault() {
  delete impl;
}

void SHashHandlerDefault::SHashReduceHashedValue(uint64_t hashed_value) {
  return impl->SHashReduceHashedValue(hashed_value);
}

void SHashHandlerDefault::SHashReduce(const ObjectRef& key, bool map_free_vars) {
  impl->SHashReduce(key, map_free_vars);
}

void SHashHandlerDefault::SHashReduceFreeVar(const runtime::Object* var, bool map_free_vars) {
  impl->SHashReduceFreeVar(var, map_free_vars);
}

bool SHashHandlerDefault::LookupHashedValue(const ObjectRef& key, uint64_t* hashed_value) {
  return impl->LookupHashedValue(key, hashed_value);
}

void SHashHandlerDefault::MarkGraphNode() {
  impl->MarkGraphNode();
}

uint64_t SHashHandlerDefault::Hash(const ObjectRef& object, bool map_free_vars) {
  return impl->Hash(object, map_free_vars);
}

void SHashHandlerDefault::DispatchSHash(const ObjectRef& key, bool map_free_vars) {
  impl->DispatchSHash(key, map_free_vars);
}

MATXSCRIPT_REGISTER_GLOBAL("node.StructuralHash")
    .set_body_typed([](const ObjectRef& object, bool map_free_vars) -> int64_t {
      uint64_t hashed_value = SHashHandlerDefault().Hash(object, map_free_vars);
      return static_cast<int64_t>(hashed_value);
    });

uint64_t StructuralHash::operator()(const ObjectRef& object) const {
  return SHashHandlerDefault().Hash(object, false);
}

}  // namespace ir
}  // namespace matxscript
