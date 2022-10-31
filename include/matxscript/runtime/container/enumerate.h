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

#include <iterator>

#include <matxscript/runtime/container/unicode.h>
#include "_item_type_traits.h"

namespace matxscript {
namespace runtime {

namespace details {

template <class Iterator, class ValueType>
using enumerate_iterator_base =
    std::iterator<typename std::iterator_traits<Iterator>::iterator_category,
                  std::pair<int64_t, ValueType>>;

template <class Iterator, class ValueType, typename TRANSFORM_TYPE>
struct enumerate_iterator : public enumerate_iterator_base<Iterator, ValueType> {
  int64_t value = 0;
  Iterator iter;
  enumerate_iterator() = default;
  enumerate_iterator(Iterator const& iter, int64_t first) : value(first), iter(iter) {
  }
  typename enumerate_iterator_base<Iterator, ValueType>::value_type operator*() const {
    return std::make_pair(value, transform_value(*iter, TRANSFORM_TYPE()));
  }
  enumerate_iterator& operator++() {
    ++value, ++iter;
    return *this;
  }
  enumerate_iterator& operator+=(int64_t n) {
    value += n, iter += n;
    return *this;
  }
  bool operator!=(enumerate_iterator const& other) const {
    return !(*this == other);
  }
  bool operator<(enumerate_iterator const& other) const {
    return iter < other.iter;
  }
  int64_t operator-(enumerate_iterator const& other) const {
    return iter - other.iter;
  }
  bool operator==(enumerate_iterator const& other) const {
    return iter == other.iter;
  }

 private:
  template <typename T_ELE>
  Unicode transform_value(T_ELE&& ele,
                          std::integral_constant<int, ITEM_TRANSFORM_TYPE_UNICODE>) const {
    return Unicode(1, ele);
  };
  template <typename T_ELE>
  T_ELE transform_value(T_ELE&& ele,
                        std::integral_constant<int, ITEM_TRANSFORM_TYPE_DEFAULT>) const {
    return ele;
  };
};

template <class Iterable>
struct enumerate
    : private std::remove_cv<typename std::remove_reference<Iterable>::type>::type,
      public enumerate_iterator<
          typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type::iterator,
          typename item_type_traits<
              typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type>::type,
          ITEM_TRANSFORM_TYPE<Iterable>> {
  using iterator = enumerate_iterator<
      typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type::iterator,
      typename item_type_traits<
          typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type>::type,
      ITEM_TRANSFORM_TYPE<Iterable>>;
  using iterator::operator*;
  iterator end_iter;

  enumerate() {
  }
  enumerate(Iterable seq, int64_t first)
      : Iterable(std::move(seq)),
        iterator(Iterable::begin(), first),
        end_iter(Iterable::end(), -1) {
  }
  iterator& begin() {
    return *this;
  }
  iterator const& begin() const {
    return *this;
  }
  iterator end() const {
    return end_iter;
  }
};

}  // namespace details

template <class Iterable>
details::enumerate<typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type>
enumerate(Iterable&& seq, int64_t first = 0L) {
  return {std::forward<Iterable>(seq), first};
}

}  // namespace runtime
}  // namespace matxscript
