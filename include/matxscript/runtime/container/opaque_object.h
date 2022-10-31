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

#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class OpaqueObjectNode;
typedef void (*FOpaqueObjectDeleter)(void* self);

// OpaqueObject is used to represent a third-object, like NativeObject but more lightweight
class OpaqueObject : public ObjectRef {
 public:
  using ContainerType = OpaqueObjectNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 public:
  explicit OpaqueObject(::matxscript::runtime::ObjectPtr<::matxscript::runtime::Object> n) noexcept
      : ObjectRef(std::move(n)) {
  }
  OpaqueObject(const OpaqueObject& other) noexcept = default;
  OpaqueObject(OpaqueObject&& other) noexcept = default;
  OpaqueObject& operator=(const OpaqueObject& other) noexcept = default;
  OpaqueObject& operator=(OpaqueObject&& other) noexcept = default;

 public:
  OpaqueObject() : OpaqueObject(-1, nullptr, nullptr) {
  }
  OpaqueObject(int64_t tag, void* ptr, FOpaqueObjectDeleter deleter);

  void* GetOpaquePtr() const;

  int64_t GetTag() const;

  unsigned char* GetInternalBufferPtr() const;
  static int GetInternalBufferSize();

  void update(int64_t tag, void* ptr, FOpaqueObjectDeleter deleter) const;
};

namespace TypeIndex {
template <>
struct type_index_traits<OpaqueObject> {
  static constexpr int32_t value = kRuntimeOpaqueObject;
};
}  // namespace TypeIndex

template <>
MATXSCRIPT_ALWAYS_INLINE OpaqueObject Any::As<OpaqueObject>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeOpaqueObject);
  return OpaqueObject(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE OpaqueObject Any::AsNoCheck<OpaqueObject>() const {
  return OpaqueObject(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

std::ostream& operator<<(std::ostream& os, OpaqueObject const& n);

}  // namespace runtime
}  // namespace matxscript
