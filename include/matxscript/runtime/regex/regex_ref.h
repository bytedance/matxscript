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

#include <matxscript/runtime/container/list_ref.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class RegexNode;

class Regex : public ObjectRef {
 public:
  using ContainerType = RegexNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

  Regex(const unicode_view& pattern,
        bool ignore_case,
        bool dotall,
        bool extended,
        bool anchored,
        bool ucp)
      : Regex(UnicodeHelper::Encode(pattern), ignore_case, dotall, extended, anchored, ucp) {
  }
  Regex(const string_view& pattern,
        bool ignore_case,
        bool dotall,
        bool extended,
        bool anchored,
        bool ucp);
  Regex(const Any& pattern, bool ignore_case, bool dotall, bool extended, bool anchored, bool ucp);

  Regex() = default;
  Regex(Regex&& other) noexcept = default;
  Regex(const Regex& other) noexcept = default;
  Regex& operator=(Regex&& other) noexcept = default;
  Regex& operator=(const Regex& other) noexcept = default;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Regex(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }

  const RegexNode* operator->() const;

  const RegexNode* get() const;

  List split(const string_view& input) const;
  List split(const unicode_view& input) const;
  List split(const Any& input) const;

  String replace(const string_view& input, const string_view& repl) const;
  inline String replace(const string_view& input, const Any& repl) const {
    return replace(input, repl.As<string_view>());
  }
  inline String replace(const Any& input, const string_view& repl) const {
    return replace(input.As<string_view>(), repl);
  }
  Unicode replace(const unicode_view& input, const unicode_view& repl) const;
  Unicode replace(const unicode_view& input, const Any& repl) const {
    return replace(input, repl.As<unicode_view>());
  }
  Unicode replace(const Any& input, const unicode_view& repl) const {
    return replace(input.As<unicode_view>(), repl);
  }
  RTValue replace(const Any& input, const Any& repl) const;

  Tuple match(const string_view& input, int64_t offset = 0) const;
  Tuple match(const unicode_view& input, int64_t offset = 0) const;
  Tuple match(const Any& input, int64_t offset = 0) const;
};

template <>
bool IsConvertible<Regex>(const Object* node);

namespace TypeIndex {
template <>
struct type_index_traits<Regex> {
  static constexpr int32_t value = kRuntimeRegex;
};
}  // namespace TypeIndex

}  // namespace runtime
}  // namespace matxscript
