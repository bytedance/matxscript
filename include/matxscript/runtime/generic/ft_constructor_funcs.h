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

#include <matxscript/runtime/ft_container.h>

namespace matxscript {
namespace runtime {

namespace Kernel_FTList {
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make() {
  return FTList<T>();
}
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make(std::initializer_list<T> init_args) {
  return FTList<T>(init_args);
}
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make(const FTList<T>& c) {
  return FTList<T>(c.begin(), c.end());
}
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make(const FTSet<T>& c) {
  return FTList<T>(c.begin(), c.end());
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make(const List& c) {
  return FTList<T>(c.begin(), c.end());
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make(const Set& c) {
  return FTList<T>(c.begin(), c.end());
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make(const Iterator& itr) {
  FTList<T> d;
  bool has_next = itr.HasNext();
  while (has_next) {
    d.push_back(itr.Next(&has_next));
  }
  return d;
}
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTList<T> make(const RTValue& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeList: {
      return make<T>(c.AsObjectViewNoCheck<List>().data());
    } break;
    case TypeIndex::kRuntimeSet: {
      return make<T>(c.AsObjectViewNoCheck<Set>().data());
    } break;
    case TypeIndex::kRuntimeIterator: {
      return make<T>(c.AsObjectViewNoCheck<Iterator>().data());
    } break;
    default: {
      return make<T>(Kernel_Iterable::make(c));
    } break;
  }
}
}  // namespace Kernel_FTList

namespace Kernel_FTSet {
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make() {
  return FTSet<T>();
}
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make(std::initializer_list<T> init_args) {
  return FTSet<T>(init_args);
}
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make(const FTSet<T>& c) {
  return FTSet<T>(c.begin(), c.end());
}
template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make(const FTList<T>& c) {
  return FTSet<T>(c.begin(), c.end());
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make(const List& c) {
  return FTSet<T>(c.begin(), c.end());
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make(const Set& c) {
  return FTSet<T>(c.begin(), c.end());
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make(const Iterator& itr) {
  FTSet<T> d;
  bool has_next = itr.HasNext();
  while (has_next) {
    d.emplace(itr.Next(&has_next));
  }
  return d;
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE FTSet<T> make(const RTValue& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return make<T>(c.AsObjectViewNoCheck<Set>().data());
    } break;
    case TypeIndex::kRuntimeList: {
      return make<T>(c.AsObjectViewNoCheck<List>().data());
    } break;
    default: {
      return make<T>(Kernel_Iterable::make(c));
    } break;
  }
}
}  // namespace Kernel_FTSet

namespace Kernel_FTDict {
template <typename K, typename V>
MATXSCRIPT_ALWAYS_INLINE FTDict<K, V> make() {
  return FTDict<K, V>{};
}
template <typename K, typename V>
MATXSCRIPT_ALWAYS_INLINE FTDict<K, V> make(
    std::initializer_list<typename FTDict<K, V>::value_type> init_args) {
  return FTDict<K, V>(init_args);
}
template <typename K, typename V>
MATXSCRIPT_ALWAYS_INLINE FTDict<K, V> make(const FTDict<K, V>& c) {
  FTDict<K, V> r;
  r.reserve(c.size());
  for (auto& value_type : c.items()) {
    r.emplace(value_type.first, value_type.second);
  }
  return r;
}
template <typename K, typename V>
MATXSCRIPT_ALWAYS_INLINE FTDict<K, V> make(const Dict& c) {
  FTDict<K, V> r;
  r.reserve(c.size());
  for (auto& value_type : c.items()) {
    r.emplace(value_type.first, value_type.second);
  }
  return r;
}
template <typename K, typename V>
MATXSCRIPT_ALWAYS_INLINE FTDict<K, V> make(const RTValue& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return make<K, V>(c.AsObjectViewNoCheck<Dict>().data());
    } break;
    default: {
      MXTHROW << "TypeError: dict(...) not support '" << c.type_name() << "'";
      return {};
    } break;
  }
}
}  // namespace Kernel_FTDict

}  // namespace runtime
}  // namespace matxscript
