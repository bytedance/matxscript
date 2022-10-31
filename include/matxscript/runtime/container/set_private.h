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

#include "_flat_hash_map.h"

#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/*! \brief Set node content in Set */
class SetNode : public Object {
 public:
  // data holder
  using _type = typename std::remove_cv<typename std::remove_reference<RTValue>::type>::type;
  // using container_type = std::unordered_set<_type, RTValueHash, RTValueEqual>;
  using container_type = ska::flat_hash_set<_type>;
  using reference = container_type::reference;
  using const_reference = container_type::const_reference;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;
  using size_type = container_type::size_type;
  using difference_type = container_type::difference_type;
  using allocator_type = container_type::allocator_type;
  using pointer = container_type::pointer;
  using const_pointer = container_type::const_pointer;

 public:
  // types
  using value_type = typename container_type::value_type;

 public:
  SetNode() : data_container() {
  }

  SetNode(std::initializer_list<value_type> init) : data_container(init) {
  }

  template <class B, class E>
  SetNode(B begin, E end) : data_container(begin, end) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeSet;
  static constexpr const char* _type_key = "runtime.Set";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SetNode, Object);

  // iterators
  MATXSCRIPT_ALWAYS_INLINE iterator begin() {
    return data_container.begin();
  }

  MATXSCRIPT_ALWAYS_INLINE const_iterator begin() const {
    return data_container.begin();
  }

  MATXSCRIPT_ALWAYS_INLINE iterator end() {
    return data_container.end();
  }

  MATXSCRIPT_ALWAYS_INLINE const_iterator end() const {
    return data_container.end();
  }

  template <typename U>
  MATXSCRIPT_ALWAYS_INLINE iterator find(const U& key) {
    return data_container.find(key);
  }

 public:
  // mutation in std::unordered_set
  MATXSCRIPT_ALWAYS_INLINE auto emplace(value_type&& item) {
    return data_container.emplace(std::move(item));
  }
  MATXSCRIPT_ALWAYS_INLINE auto emplace(const value_type& item) {
    return data_container.emplace(item);
  }

  MATXSCRIPT_ALWAYS_INLINE void add(value_type&& item) {
    data_container.emplace(std::move(item));
  }
  MATXSCRIPT_ALWAYS_INLINE void add(const value_type& item) {
    data_container.emplace(item);
  }

  MATXSCRIPT_ALWAYS_INLINE void clear() {
    data_container.clear();
  }

  MATXSCRIPT_ALWAYS_INLINE void reserve(int64_t new_size) {
    if (new_size > 0) {
      data_container.reserve(static_cast<size_t>(new_size));
    }
  }

 public:
  // const methods in std::unordered_map
  MATXSCRIPT_ALWAYS_INLINE size_t size() const {
    return data_container.size();
  }

  MATXSCRIPT_ALWAYS_INLINE int64_t bucket_count() const {
    return data_container.bucket_count();
  }

  MATXSCRIPT_ALWAYS_INLINE bool empty() const {
    return data_container.empty();
  }

  template <typename U>
  MATXSCRIPT_ALWAYS_INLINE bool contains(const U& key) const {
    return data_container.find(key) != data_container.end();
  }

 private:
  container_type data_container;

  // Reference class
  friend class Set;
  friend class SetNodeTrait;
};

}  // namespace runtime
}  // namespace matxscript
