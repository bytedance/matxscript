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

#include <cstddef>
#include <functional>
#include <memory>

#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Generic iterator, like python iter
 *****************************************************************************/
class IteratorNode;

class Iterator : public ObjectRef {
 public:
  using ContainerType = IteratorNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 public:
  // operators
  bool HasNext() const;
  RTValue Next() const;
  RTValue Next(bool* has_next) const;
  RTView NextView(bool* has_next, RTValue* holder_or_null) const;
  int64_t Distance() const;
  Iterator() noexcept = default;
  // copy and assign
  Iterator(const Iterator& other) noexcept = default;
  Iterator(Iterator&& other) noexcept = default;
  Iterator& operator=(const Iterator& other) noexcept = default;
  Iterator& operator=(Iterator&& other) noexcept = default;
  explicit Iterator(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }

  // Node
  IteratorNode* GetMutableNode() const;

 public:
  static Iterator MakeGenericIterator(RTValue container,
                                      std::function<bool()> has_next,
                                      std::function<RTValue()> next,
                                      std::function<RTValue(bool*)> next_and_check);

  static Iterator MakeGenericIterator(const Any& container);

  static Iterator MakeItemsIterator(const Any& container);

  static bool all_items_equal(const Iterator& lhs, const Iterator& rhs);
};

namespace TypeIndex {
template <>
struct type_index_traits<Iterator> {
  static constexpr int32_t value = kRuntimeIterator;
};
}  // namespace TypeIndex

template <>
bool IsConvertible<Iterator>(const Object* node);

// RTValue converter
template <>
MATXSCRIPT_ALWAYS_INLINE Iterator Any::As<Iterator>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeIterator);
  return Iterator(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE Iterator Any::AsNoCheck<Iterator>() const {
  return Iterator(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

std::ostream& operator<<(std::ostream& os, Iterator const& n);

}  // namespace runtime
}  // namespace matxscript
