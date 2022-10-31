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

#include "./itertor_ref.h"

#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>

#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * IteratorNode interface
 *
 * Each container should implement its own IteratorNode
 * if you want to improve iterator performance,
 *****************************************************************************/

class IteratorNode : public Object {
 public:
  explicit IteratorNode() = default;
  ~IteratorNode() = default;

  virtual uint64_t HashCode() const = 0;

  virtual bool HasNext() const = 0;
  virtual RTValue Next() = 0;
  virtual RTValue Next(bool* has_next) = 0;
  virtual RTView NextView(bool* has_next, RTValue* holder_or_null) = 0;
  virtual int64_t Distance() const = 0;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeIterator;
  static constexpr const char* _type_key = "Iterator";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IteratorNode, Object);

 public:
  friend class IteratorNodeTrait;
};

/******************************************************************************
 * Generic iterator, like python iter
 *****************************************************************************/
template <typename ResultType>
inline RTValue IteratorSetRuntimeValue(ResultType result) {
  return (*result);
}

class GenericIterator {
 public:
  inline bool HasNext() const {
    return has_next_();
  }
  inline RTValue Next() {
    ++iter_cou_;
    return next_();
  }
  inline RTValue Next(bool* has_next) {
    ++iter_cou_;
    return next_and_checker_(has_next);
  }
  GenericIterator() {
    has_next_ = []() -> bool { return false; };
    next_ = []() -> RTValue { return RTValue(); };
  }
  GenericIterator(std::function<bool()> has_next,
                  std::function<RTValue()> next,
                  std::function<RTValue(bool*)> next_and_check)
      : iter_cou_(0),
        has_next_(std::move(has_next)),
        next_(std::move(next)),
        next_and_checker_(std::move(next_and_check)) {
  }
  template <class ITERATOR_TYPE>
  explicit GenericIterator(ITERATOR_TYPE&& iter_begin, ITERATOR_TYPE&& iter_end) {
    iter_cou_ = 0;
    auto iterator_ptr = std::make_shared<ITERATOR_TYPE>(std::forward<ITERATOR_TYPE>(iter_begin));
    auto* iter_c = static_cast<ITERATOR_TYPE*>(iterator_ptr.get());
    has_next_ = [iter_c, iterator_ptr, iter_end]() -> bool { return *iter_c != iter_end; };
    next_ = [iter_c, iter_end]() -> RTValue {
      RTValue r = IteratorSetRuntimeValue<ITERATOR_TYPE>(*iter_c);
      ++(*iter_c);
      return r;
    };
    next_and_checker_ = [iter_c, iter_end](bool* has_next) -> RTValue {
      RTValue r = IteratorSetRuntimeValue<ITERATOR_TYPE>(*iter_c);
      ++(*iter_c);
      *has_next = (*iter_c != iter_end);
      return r;
    };
  };

 private:
  int64_t iter_cou_;
  std::function<bool()> has_next_;
  std::function<RTValue()> next_;
  std::function<RTValue(bool*)> next_and_checker_;
  friend class GenericIteratorNode;
  friend class IteratorNodeTrait;
};

/******************************************************************************
 * Generic iterator Object
 *****************************************************************************/
class GenericIteratorNode : public IteratorNode {
 public:
  template <typename CONTAINER_TYPE, typename ITERATOR_TYPE>
  explicit GenericIteratorNode(CONTAINER_TYPE&& container,
                               ITERATOR_TYPE&& iter,
                               ITERATOR_TYPE&& iter_end)
      : container_(std::forward<CONTAINER_TYPE>(container)),
        iterator_(std::forward<ITERATOR_TYPE>(iter), std::forward<ITERATOR_TYPE>(iter_end)) {
  }
  explicit GenericIteratorNode(RTValue container,
                               std::function<bool()> has_next,
                               std::function<RTValue()> next,
                               std::function<RTValue(bool*)> next_and_check)
      : container_(std::move(container)),
        iterator_(std::move(has_next), std::move(next), std::move(next_and_check)) {
  }
  ~GenericIteratorNode() = default;

  uint64_t HashCode() const override {
    return reinterpret_cast<uint64_t>(container_.ptr<Object>());
  }

  bool HasNext() const override {
    return iterator_.HasNext();
  }
  RTValue Next() override {
    return iterator_.Next();
  }
  RTValue Next(bool* has_next) override {
    return iterator_.Next(has_next);
  }
  RTView NextView(bool* has_next, RTValue* holder_or_null) override {
    *holder_or_null = iterator_.Next(has_next);
    return *holder_or_null;
  }

  int64_t Distance() const override {
    return -1;
  }

 public:
  RTValue container_;
  GenericIterator iterator_;
  friend class IteratorNodeTrait;
};

template <class CONTAINER_TYPE, class ITERATOR_TYPE>
static inline Iterator MakeGenericIterator(CONTAINER_TYPE&& container,
                                           ITERATOR_TYPE&& iter_begin,
                                           ITERATOR_TYPE&& iter_end) {
  auto data = make_object<GenericIteratorNode>(std::forward<CONTAINER_TYPE>(container),
                                               std::forward<ITERATOR_TYPE>(iter_begin),
                                               std::forward<ITERATOR_TYPE>(iter_end));
  return Iterator(std::move(data));
}

}  // namespace runtime
}  // namespace matxscript
