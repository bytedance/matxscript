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

#include <string>

#include <matxscript/runtime/container/itertor_ref.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class FileNode;

class File : public ObjectRef {
 public:
  using ContainerType = FileNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

  // constructors
  /*!
   * \brief constructor
   */
  explicit File(const Unicode& path,
                const Unicode& mode = U"r",
                const Unicode& encoding = U"utf-8");

  /*!
   * \brief move constructor
   * \param other source
   */
  File(File&& other) noexcept;

  /*!
   * \brief copy constructor
   */
  File(const File& other) noexcept = default;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit File(ObjectPtr<Object> n) noexcept;

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  File& operator=(File&& other) noexcept;

  /*!
   * \brief copy assign operator
   */
  File& operator=(const File& other) noexcept = default;

  const FileNode* operator->() const;
  const FileNode* get() const;

 public:
  // method for python
  bool HasNext() const;
  string_view path() const;
  String ReadString(int64_t size = -1) const;
  Unicode ReadUnicode(int64_t size = -1) const;
  RTValue Read(int64_t size = -1) const;
  String ReadLineString() const;
  Unicode ReadLineUnicode() const;
  List ReadLines() const;
  RTValue Next() const;
  RTValue Next(bool* has_next) const;
  RTView NextView(bool* has_next, RTValue* holder_or_null) const;
  void close() const;
};

namespace TypeIndex {
template <>
struct type_index_traits<File> {
  static constexpr int32_t value = kRuntimeFile;
};
}  // namespace TypeIndex

template <>
bool IsConvertible<File>(const Object* node);

std::ostream& operator<<(std::ostream& os, File const& n);

}  // namespace runtime
}  // namespace matxscript
