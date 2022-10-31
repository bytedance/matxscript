// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 * https://github.com/apache/tvm/blob/v0.7/include/tvm/runtime/container.h
 * with changes applied:
 * - rename namespace
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
#pragma once

#include <iterator>
#include <type_traits>

#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

template <typename Converter, typename IterType>
class IteratorAdapter {
 public:
  using difference_type = typename std::iterator_traits<IterType>::difference_type;
  using value_type = typename Converter::return_type;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = typename std::iterator_traits<IterType>::iterator_category;

  explicit IteratorAdapter(IterType iter) : iter_(iter) {
  }
  IteratorAdapter& operator++() {
    ++iter_;
    return *this;
  }
  IteratorAdapter& operator--() {
    --iter_;
    return *this;
  }
  IteratorAdapter operator++(int) {
    IteratorAdapter copy = *this;
    ++iter_;
    return copy;
  }
  IteratorAdapter operator--(int) {
    IteratorAdapter copy = *this;
    --iter_;
    return copy;
  }
  IteratorAdapter& operator+=(int offset) {
    iter_ += offset;
    return *this;
  }

  IteratorAdapter operator+(difference_type offset) const {
    return IteratorAdapter(iter_ + offset);
  }

  template <typename T = IteratorAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type inline
  operator-(const IteratorAdapter& rhs) const {
    return iter_ - rhs.iter_;
  }

  bool operator==(const IteratorAdapter& other) const {
    return iter_ == other.iter_;
  }
  bool operator!=(const IteratorAdapter& other) const {
    return !(*this == other);
  }
  const value_type operator*() const {
    static Converter conv;
    return conv(*iter_);
  }

 private:
  IterType iter_;
};

template <typename Converter, typename IterType>
class ReverseIteratorAdapter {
 public:
  using difference_type = typename std::iterator_traits<IterType>::difference_type;
  using value_type = typename Converter::return_type;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = typename std::iterator_traits<IterType>::iterator_category;

  explicit ReverseIteratorAdapter(IterType iter) : iter_(iter) {
  }
  ReverseIteratorAdapter& operator++() {
    --iter_;
    return *this;
  }
  ReverseIteratorAdapter& operator--() {
    ++iter_;
    return *this;
  }
  ReverseIteratorAdapter& operator++(int) {
    ReverseIteratorAdapter copy = *this;
    --iter_;
    return copy;
  }
  ReverseIteratorAdapter& operator--(int) {
    ReverseIteratorAdapter copy = *this;
    ++iter_;
    return copy;
  }
  ReverseIteratorAdapter& operator+=(int offset) {
    iter_ += offset;
    return *this;
  }
  ReverseIteratorAdapter operator+(difference_type offset) const {
    return ReverseIteratorAdapter(iter_ - offset);
  }

  template <typename T = ReverseIteratorAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type inline
  operator-(const ReverseIteratorAdapter& rhs) const {
    return rhs.iter_ - iter_;
  }

  bool operator==(ReverseIteratorAdapter other) const {
    return iter_ == other.iter_;
  }
  bool operator!=(ReverseIteratorAdapter other) const {
    return !(*this == other);
  }
  const value_type operator*() const {
    static Converter conv;
    return conv(*iter_);
  }

 private:
  IterType iter_;
};

}  // namespace runtime
}  // namespace matxscript
