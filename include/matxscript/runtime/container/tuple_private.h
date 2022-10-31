// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 * https://github.com/apache/tvm/blob/v0.7/include/tvm/runtime/container.h
 * with changes applied:
 * - rename namespace
 * - implement some tuple methods
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

#include "inplace_array_base.h"

#include <initializer_list>

#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/*! \brief An object representing a structure or enumeration. */
class TupleNode : public Object, public InplaceArrayBase<TupleNode, RTValue> {
 public:
  using value_type = RTValue;
  using pointer = RTValue*;
  using const_pointer = const RTValue*;
  using reference = RTValue&;
  using const_reference = const RTValue&;
  using const_iterator = const RTValue*;
  using iterator = const_iterator;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = const_reverse_iterator;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  /*! \brief Number of fields in the ADT object. */
  size_t size;
  // The fields of the structure follows directly in memory.

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeTuple;
  static constexpr const char* _type_key = "runtime.Tuple";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TupleNode, Object);

 public:
  // iterators
  MATXSCRIPT_ALWAYS_INLINE iterator begin() {
    return reinterpret_cast<value_type*>(AddressOf(0));
  }

  MATXSCRIPT_ALWAYS_INLINE const_iterator begin() const {
    return reinterpret_cast<value_type*>(AddressOf(0));
  }

  MATXSCRIPT_ALWAYS_INLINE iterator end() {
    return reinterpret_cast<value_type*>(AddressOf(size));
  }

  MATXSCRIPT_ALWAYS_INLINE const_iterator end() const {
    return reinterpret_cast<value_type*>(AddressOf(size));
  }

  MATXSCRIPT_ALWAYS_INLINE reverse_iterator rbegin() {
    return reverse_iterator(end());
  }

  MATXSCRIPT_ALWAYS_INLINE const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  MATXSCRIPT_ALWAYS_INLINE reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  MATXSCRIPT_ALWAYS_INLINE const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  MATXSCRIPT_ALWAYS_INLINE value_type* data() {
    return reinterpret_cast<value_type*>(AddressOf(0));
  }

  MATXSCRIPT_ALWAYS_INLINE const value_type* data() const {
    return reinterpret_cast<value_type*>(AddressOf(0));
  }

  static ObjectPtr<TupleNode> MakeNones(size_t n);

 private:
  /*!
   * \return The number of elements in the array.
   */
  size_t GetSize() const {
    return size;
  }

  /*!
   * \brief Initialize the elements in the array.
   *
   * \tparam Iterator Iterator type of the array.
   * \param begin The begin iterator.
   * \param end The end iterator.
   */
  template <typename Iterator>
  void Init(Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    this->size = 0;
    auto it = begin;
    for (size_t i = 0; i < num_elems; ++i) {
      InplaceArrayBase::EmplaceInit(i, *it++);
      // Only increment size after the initialization succeeds
      this->size++;
    }
  }

  friend class Tuple;
  friend InplaceArrayBase<TupleNode, RTValue>;
};

}  // namespace runtime
}  // namespace matxscript
