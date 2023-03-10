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
 * \file src/node/structural_equal.cc
 */
#include <matxscript/ir/_base/structural_equal.h>

#include <unordered_map>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace runtime;

// Define the dispatch function here since primary user is in this file.
bool ReflectionVTable::SEqualReduce(const Object* self,
                                    const Object* other,
                                    SEqualReducer equal) const {
  uint32_t tindex = self->type_index();
  if (tindex >= fsequal_reduce_.size() || fsequal_reduce_[tindex] == nullptr) {
    MXLOG(FATAL) << "TypeError: SEqualReduce of " << self->GetTypeKey()
                 << " is not registered via MATXSCRIPT_REGISTER_NODE_TYPE."
                 << " Did you forget to set _type_has_method_sequal_reduce=true?";
  }
  return fsequal_reduce_[tindex](self, other, equal);
}

/*!
 * \brief A non recursive stack based SEqual handler that can remaps vars.
 *
 *  This handler pushs the Object equality cases into a stack, and
 *  traverses the stack to expand the necessary children that need to be checked.
 *
 *  The order of SEqual being called is the same as the order as if we
 *  eagerly do recursive calls in SEqualReduce.
 */
class RemapVarSEqualHandler : public SEqualReducer::Handler {
 public:
  explicit RemapVarSEqualHandler(bool assert_mode) : assert_mode_(assert_mode) {
  }

  bool SEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars) final {
    // We cannot use check lhs.same_as(rhs) to check equality.
    // if we choose to enable var remapping.
    //
    // Counter example below (%x, %y) are shared vars
    // between the two functions(possibly before/after rewriting).
    //
    // - function0: fn (%x, %y) { %x + %y }
    // - function1. fn (%y, %x) { %x + %y }
    //
    // Because we choose to enable var remapping,
    // %x is mapped to %y, and %y is mapped to %x,
    // the body of the function no longer means the same thing.
    //
    // Take away: We can either choose only compare Var by address,
    // in which case we can use same_as for quick checking,
    // or we have to run deep comparison and avoid to use same_as checks.
    auto run = [=]() {
      if (!lhs.defined() && !rhs.defined())
        return true;
      if (!lhs.defined() && rhs.defined())
        return false;
      if (!rhs.defined() && lhs.defined())
        return false;
      if (lhs->type_index() != rhs->type_index())
        return false;
      auto it = equal_map_lhs_.find(lhs);
      if (it != equal_map_lhs_.end()) {
        return it->second.same_as(rhs);
      }
      if (equal_map_rhs_.count(rhs))
        return false;
      // need to push to pending tasks in this case
      pending_tasks_.emplace_back(Task(lhs, rhs, map_free_vars));
      return true;
    };
    return CheckResult(run(), lhs, rhs);
  }

  void MarkGraphNode() final {
    // need to push to pending tasks in this case
    MXCHECK(!allow_push_to_stack_ && !task_stack_.empty());
    task_stack_.back().graph_equal = true;
  }

  ObjectRef MapLhsToRhs(const ObjectRef& lhs) final {
    auto it = equal_map_lhs_.find(lhs);
    if (it != equal_map_lhs_.end())
      return it->second;
    return ObjectRef(nullptr);
  }

  // Function that implements actual equality check.
  bool Equal(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars) {
    if (!lhs.defined() && !rhs.defined())
      return true;
    task_stack_.clear();
    pending_tasks_.clear();
    equal_map_lhs_.clear();
    equal_map_rhs_.clear();
    if (!SEqualReduce(lhs, rhs, map_free_vars))
      return false;
    MXCHECK_EQ(pending_tasks_.size(), 1U);
    MXCHECK(allow_push_to_stack_);
    task_stack_.emplace_back(std::move(pending_tasks_.back()));
    pending_tasks_.clear();
    return RunTasks();
  }

 protected:
  // Check the result.
  bool CheckResult(bool result, const ObjectRef& lhs, const ObjectRef& rhs) {
    if (assert_mode_ && !result) {
      MXLOG(FATAL) << "ValueError: StructuralEqual check failed, caused by\n"
                   << "lhs = " << lhs << "\nrhs = " << rhs;
    }
    return result;
  }
  /*!
   * \brief Run tasks until the stack reaches the stack begin
   * \param stack_begin The expected beginning of the stack.
   * \return The checks we encountered throughout the process.
   */
  bool RunTasks() {
    while (task_stack_.size() != 0) {
      // Caution: entry becomes invalid when the stack changes
      auto& entry = task_stack_.back();

      if (entry.children_expanded) {
        // When all the children has expanded and visited.
        // This means all the condition checks for
        // the current entry has been passed
        // We can safely mark lhs and rhs as equal to each other.
        auto it = equal_map_lhs_.find(entry.lhs);
        if (it != equal_map_lhs_.end()) {
          MXCHECK(it->second.same_as(entry.rhs));
        }
        // create the map if the quality is graph equal.
        if (entry.graph_equal) {
          equal_map_lhs_[entry.lhs] = entry.rhs;
          equal_map_rhs_[entry.rhs] = entry.lhs;
        }
        task_stack_.pop_back();
      } else {
        // mark before expand
        // Important: because entry becomes invalid when stack changes.
        entry.children_expanded = true;
        // Expand the objects
        // The SEqual of the object can call into this->SEqualReduce
        // which populates the pending tasks.
        MXCHECK_EQ(pending_tasks_.size(), 0U);
        allow_push_to_stack_ = false;
        if (!DispatchSEqualReduce(entry.lhs, entry.rhs, entry.map_free_vars))
          return false;
        allow_push_to_stack_ = true;
        // Push pending tasks in reverse order, so earlier tasks get to
        // expand first in the stack
        while (pending_tasks_.size() != 0) {
          task_stack_.emplace_back(std::move(pending_tasks_.back()));
          pending_tasks_.pop_back();
        }
      }
    }
    return true;
  }

  // The default equal as registered in the structural equal vtable.
  bool DispatchSEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars) {
    auto compute = [=]() {
      MXCHECK(lhs.defined() && rhs.defined() && lhs->type_index() == rhs->type_index());
      // skip entries that already have equality maps.
      auto it = equal_map_lhs_.find(lhs);
      if (it != equal_map_lhs_.end()) {
        return it->second.same_as(rhs);
      }
      if (equal_map_rhs_.count(rhs))
        return false;
      // Run reduce check for free nodes.
      return vtable_->SEqualReduce(lhs.get(), rhs.get(), SEqualReducer(this, map_free_vars));
    };
    return CheckResult(compute(), lhs, rhs);
  }

 private:
  /*! \brief Pending reduce tasks. */
  struct Task {
    /*! \brief The lhs operand to be compared. */
    ObjectRef lhs;
    /*! \brief The rhs operand to be compared. */
    ObjectRef rhs;
    /*! \brief The map free var argument. */
    bool map_free_vars;
    /*! \brief Whether the children has been expanded via SEqualReduce */
    bool children_expanded{false};
    /*! \brief whether the task is about graph equality(need remap). */
    bool graph_equal{false};

    Task() = default;
    Task(ObjectRef lhs, ObjectRef rhs, bool map_free_vars)
        : lhs(lhs), rhs(rhs), map_free_vars(map_free_vars) {
    }
  };
  // list of pending tasks to be pushed to the stack.
  std::vector<Task> pending_tasks_;
  // Internal task stack to executed the task.
  std::vector<Task> task_stack_;
  // Whether we allow push to stack.
  bool allow_push_to_stack_{true};
  //  If in assert mode, must return true, and will throw error otherwise.
  bool assert_mode_{false};
  // reflection vtable
  ReflectionVTable* vtable_ = ReflectionVTable::Global();
  // map from lhs to rhs
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> equal_map_lhs_;
  // map from rhs to lhs
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> equal_map_rhs_;
};

MATXSCRIPT_REGISTER_GLOBAL("runtime.StructuralEqual")
    .set_body_typed(
        [](const ObjectRef& lhs, const ObjectRef& rhs, bool assert_mode, bool map_free_vars) {
          return RemapVarSEqualHandler(assert_mode).Equal(lhs, rhs, map_free_vars);
        });

bool StructuralEqual::operator()(const ObjectRef& lhs, const ObjectRef& rhs) const {
  return RemapVarSEqualHandler(false).Equal(lhs, rhs, false);
}

bool StructuralEqual::operator()(const RTValue& lhs, const RTValue& rhs) const {
  // quick specialization for Array to reduce amount of recursion
  // depth as array comparison is pretty common.
  if (lhs.type_code() != rhs.type_code())
    return false;
  int32_t type_code = lhs.type_code();
  switch (type_code) {
    case TypeIndex::kRuntimeNullptr: {
      return true;
    } break;
    case TypeIndex::kRuntimeOpaqueHandle: {
      return lhs.As<void*>() == rhs.As<void*>();
    } break;
    case TypeIndex::kRuntimeInteger: {
      return this->operator()(lhs.As<int64_t>(), rhs.As<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      return this->operator()(lhs.As<double>(), rhs.As<double>());
    } break;
    case TypeIndex::kRuntimeString: {
      return this->operator()(lhs.AsNoCheck<string_view>(), rhs.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return this->operator()(lhs.AsNoCheck<unicode_view>(), rhs.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeStringRef: {
      return lhs.ptr<StringNode>()->data_container == rhs.ptr<StringNode>()->data_container;
    } break;
    default: {
      return this->operator()(lhs.As<ObjectRef>(), rhs.As<ObjectRef>());
    } break;
  }
  return false;
}

bool SEqualReducer::operator()(const RTValue& lhs, const RTValue& rhs) const {
  if (lhs.type_code() != rhs.type_code())
    return false;
  int32_t type_code = lhs.type_code();
  switch (type_code) {
    case TypeIndex::kRuntimeNullptr: {
      return true;
    } break;
    case TypeIndex::kRuntimeOpaqueHandle: {
      return lhs.As<void*>() == rhs.As<void*>();
    } break;
    case TypeIndex::kRuntimeInteger: {
      return this->operator()(lhs.As<int64_t>(), rhs.As<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      return this->operator()(lhs.As<double>(), rhs.As<double>());
    } break;
    case TypeIndex::kRuntimeString: {
      return this->operator()(lhs.AsNoCheck<string_view>(), rhs.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return this->operator()(lhs.AsNoCheck<unicode_view>(), rhs.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeStringRef: {
      return lhs.ptr<StringNode>()->data_container == rhs.ptr<StringNode>()->data_container;
    } break;
    default: {
      return this->operator()(lhs.As<ObjectRef>(), rhs.As<ObjectRef>());
    } break;
  }
  return false;
}

}  // namespace ir
}  // namespace matxscript
