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
#include "string_view.h"

#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class KwargsNode : public Object {
 public:
  // data holder
  using _key_type = string_view;
  using _mapped_type = RTValue;
  using container_type = ska::flat_hash_map<_key_type, _mapped_type>;

 public:
  // types
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using size_type = typename container_type::size_type;
  using difference_type = typename container_type::difference_type;
  using value_type = typename container_type::value_type;
  using allocator_type = typename container_type::allocator_type;
  using pointer = typename container_type::pointer;
  using const_pointer = typename container_type::const_pointer;
  using key_type = typename container_type::key_type;
  using mapped_type = typename container_type::mapped_type;

 public:
  KwargsNode() : data_container() {
  }
  KwargsNode(std::initializer_list<value_type> init) : data_container(init) {
  }

  template <class B, class E>
  KwargsNode(B begin, E end) : data_container(begin, end) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeKwargs;
  static constexpr const char* _type_key = "runtime.Kwargs";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(KwargsNode, Object);

 public:
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

 public:
  template <class KEY_T>
  MATXSCRIPT_ALWAYS_INLINE const_iterator find(const KEY_T& key) const {
    return data_container.find(key);
  }

  template <class KEY_T>
  MATXSCRIPT_ALWAYS_INLINE iterator find(const KEY_T& key) {
    return data_container.find(key);
  }

  template <typename It>
  MATXSCRIPT_ALWAYS_INLINE void insert(It begin, It end) {
    return data_container.insert(begin, end);
  }

  MATXSCRIPT_ALWAYS_INLINE void insert(std::initializer_list<value_type> il) {
    return data_container.insert(il);
  }

  MATXSCRIPT_ALWAYS_INLINE mapped_type& operator[](key_type key) {
    return data_container[std::move(key)];
  }

  template <typename Key, typename... Args>
  MATXSCRIPT_ALWAYS_INLINE std::pair<iterator, bool> emplace(Key&& key, Args&&... args) {
    return data_container.emplace(std::forward<Key>(key), std::forward<Args>(args)...);
  }

  MATXSCRIPT_ALWAYS_INLINE void clear() {
    data_container.clear();
  }

  MATXSCRIPT_ALWAYS_INLINE void reserve(int64_t new_size) {
    if (new_size > 0) {
      data_container.reserve(new_size);
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

  template <class KEY_U>
  MATXSCRIPT_ALWAYS_INLINE bool contains(KEY_U const& key) const {
    return data_container.find(key) != data_container.end();
  }

 private:
  container_type data_container;

  // Reference class
  friend class Kwargs;
  friend class KwargsNodeTrait;
};

}  // namespace runtime
}  // namespace matxscript
