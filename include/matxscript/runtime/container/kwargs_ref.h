// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of iterator_adaptator is inspired by pythran.
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

#include "string_view.h"
#include "unicode_view.h"

#include <initializer_list>
#include <vector>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class KwargsNode;

class Kwargs : public ObjectRef {
 public:
  using ContainerType = KwargsNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance
  using value_type = std::pair<string_view, RTValue>;

 public:
  // constructors
  /*!
   * \brief default constructor
   */
  Kwargs();

  /*!
   * \brief move constructor
   * \param other source
   */
  Kwargs(Kwargs&& other) noexcept = default;

  /*!
   * \brief copy constructor
   * \param other source
   */
  Kwargs(const Kwargs& other) noexcept = default;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Kwargs(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }

  /*!
   * \brief constructor from initializer Kwargs
   * \param init The initializer Kwargs
   */
  Kwargs(std::initializer_list<value_type> init);

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Kwargs& operator=(Kwargs&& other) noexcept = default;

  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Kwargs& operator=(const Kwargs& other) noexcept = default;

 public:
  // method for python
  RTValue& get_item(string_view key) const;

  // mutation in std::unordered_map
  RTValue& operator[](string_view key) const;

  // const methods in std::unordered_map
  int64_t size() const;

  bool empty() const;

  bool contains(const string_view& key) const;

  string_view diff(string_view* args, size_t num_args) const;
};

struct KwargsUnpackHelper {
  KwargsUnpackHelper(string_view func_name,
                     string_view* arg_names,
                     size_t num_args,
                     Any* defaults,
                     size_t num_defaults)
      : func_name_(func_name),
        arg_names_(arg_names),
        num_args_(num_args),
        default_args_(defaults),
        num_default_args_(num_defaults) {
  }

  void unpack(RTView* pos_args, PyArgs original_args) const;
  void unpack(RTView* pos_args, MATXScriptAny* original_args, int num_original_args) const;

 private:
  string_view func_name_;
  string_view* arg_names_;
  size_t num_args_;
  Any* default_args_;
  size_t num_default_args_;
};

namespace TypeIndex {
template <>
struct type_index_traits<Kwargs> {
  static constexpr int32_t value = kRuntimeKwargs;
};
}  // namespace TypeIndex

// iterators
template <>
MATXSCRIPT_ALWAYS_INLINE Kwargs Any::As<Kwargs>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeKwargs);
  return Kwargs(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE Kwargs Any::AsNoCheck<Kwargs>() const {
  return Kwargs(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

std::ostream& operator<<(std::ostream& os, Kwargs const& n);  // namespace runtime

template <>
bool IsConvertible<Kwargs>(const Object* node);

}  // namespace runtime
}  // namespace matxscript
