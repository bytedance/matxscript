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

#include "generator.h"
#include "yielder.h"

#include <cstddef>
#include <iterator>

#include <matxscript/runtime/container/itertor_ref.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

// BaseGenerator is used for codegen only
template <typename ITEM_VALUE_TYPE>
struct BaseGenerator : ::matxscript::runtime::Yielder {
  static_assert(std::is_same<ITEM_VALUE_TYPE, bool>::value ||
                    std::is_same<ITEM_VALUE_TYPE, int32_t>::value ||
                    std::is_same<ITEM_VALUE_TYPE, int64_t>::value ||
                    std::is_same<ITEM_VALUE_TYPE, float>::value ||
                    std::is_same<ITEM_VALUE_TYPE, double>::value ||
                    std::is_same<ITEM_VALUE_TYPE, RTValue>::value,
                "Generator only support int32/int64/float/double/bool/RTValue");
  typedef GeneratorIterator<BaseGenerator> iterator;
  typedef ITEM_VALUE_TYPE value_type;
  typedef ITEM_VALUE_TYPE result_type;

  BaseGenerator() : Yielder() {
  }
  virtual ~BaseGenerator() = default;
  void operator++() {
    next();
  }
  virtual result_type operator*() const = 0;
  virtual result_type next() = 0;
};

// GeneratorAdapter is used for codegen only
template <typename ITEM_VALUE_TYPE>
struct GeneratorAdapter {
  typedef BaseGenerator<ITEM_VALUE_TYPE> IGenerator;
  typedef GeneratorIterator<GeneratorAdapter> iterator;
  typedef ITEM_VALUE_TYPE value_type;
  typedef ITEM_VALUE_TYPE result_type;

  explicit GeneratorAdapter(std::shared_ptr<IGenerator> generator)
      : generator_(std::move(generator)) {
  }
  template <typename GENERATOR_CHILD_TYPE>
  explicit GeneratorAdapter(GENERATOR_CHILD_TYPE* ptr) {
    static_assert(std::is_base_of<IGenerator, GENERATOR_CHILD_TYPE>::value,
                  "not inherit BaseGenerator");
    generator_ = std::shared_ptr<IGenerator>(static_cast<IGenerator*>(ptr));
  }
  explicit GeneratorAdapter() : generator_(nullptr) {
  }
  virtual ~GeneratorAdapter() = default;
  GeneratorAdapter(GeneratorAdapter const&) = default;
  GeneratorAdapter& operator=(GeneratorAdapter const&) = default;
  GeneratorAdapter(GeneratorAdapter&&) noexcept = default;
  GeneratorAdapter& operator=(GeneratorAdapter&&) noexcept = default;
  void operator++() {
    return generator_->operator++();
  }
  result_type operator*() const {
    return generator_->operator*();
  }
  bool operator!=(GeneratorAdapter const& other) const {
    return !operator==(other);
  }
  bool operator==(GeneratorAdapter const& other) const {
    if (!generator_ && !other.generator_) {
      return true;
    } else if (generator_ && other.generator_) {
      return generator_ == other.generator_ && generator_->operator==(*other.generator_);
    } else if (generator_) {
      return generator_->GetState() == -1;
    } else {
      return other.generator_->GetState() == -1;
    }
  }
  GeneratorIterator<GeneratorAdapter> begin() {
    next();
    return GeneratorIterator<GeneratorAdapter>(*this);
  }
  GeneratorIterator<GeneratorAdapter> end() {
    return GeneratorIterator<GeneratorAdapter>();
  }
  result_type next() {
    MXCHECK(generator_) << "generator_ is null";
    return generator_->next();
  }

  inline void SetState(int64_t stat) {
    if (generator_) {
      generator_->SetState(stat);
    }
  }

  inline int64_t GetState() const {
    if (generator_) {
      return generator_->GetState();
    } else {
      return -1;
    }
  }
  std::shared_ptr<IGenerator> generator_;
};

#define DECLARE_GENERATOR_OBJECT_REF(Name, ResultType)                                \
  class Name##Node;                                                                   \
  class Name : public ObjectRef {                                                     \
   public:                                                                            \
    using ContainerType = Name##Node;                                                 \
    static constexpr bool _type_is_nullable = false;                                  \
                                                                                      \
   public:                                                                            \
    typedef GeneratorAdapter<ResultType> GAdapter;                                    \
    typedef GAdapter::iterator iterator;                                              \
    typedef iterator::value_type value_type;                                          \
                                                                                      \
    Name() = default;                                                                 \
    explicit Name(::matxscript::runtime::ObjectPtr<::matxscript::runtime::Object> n)  \
        : ObjectRef(n) {                                                              \
    }                                                                                 \
    Name(const Name& other) noexcept = default;                                       \
    Name(Name&& other) noexcept = default;                                            \
    Name& operator=(const Name& other) noexcept = default;                            \
    Name& operator=(Name&& other) noexcept = default;                                 \
    explicit Name(std::shared_ptr<BaseGenerator<ResultType>> generator);              \
    explicit Name(GAdapter generator);                                                \
    void operator++() const {                                                         \
      return GetGenerator()->operator++();                                            \
    }                                                                                 \
    ResultType operator*() const {                                                    \
      return GetGenerator()->operator*();                                             \
    }                                                                                 \
    Iterator iter() const;                                                            \
    GAdapter::iterator begin() const {                                                \
      return GetGenerator()->begin();                                                 \
    }                                                                                 \
    GAdapter::iterator end() const {                                                  \
      return GetGenerator()->end();                                                   \
    }                                                                                 \
    ResultType next() const {                                                         \
      return GetGenerator()->next();                                                  \
    }                                                                                 \
                                                                                      \
   private:                                                                           \
    GAdapter* GetGenerator() const;                                                   \
  };                                                                                  \
  template <>                                                                         \
  bool IsConvertible<Name>(const Object* node);                                       \
  template <>                                                                         \
  MATXSCRIPT_ALWAYS_INLINE Name Any::As<Name>() const {                               \
    MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntime##Name); \
    return Name(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));    \
  }                                                                                   \
  template <>                                                                         \
  MATXSCRIPT_ALWAYS_INLINE Name Any::AsNoCheck<Name>() const {                        \
    return Name(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));    \
  }                                                                                   \
  std::ostream& operator<<(std::ostream& os, Name const& n)

DECLARE_GENERATOR_OBJECT_REF(BoolGenerator, bool);
DECLARE_GENERATOR_OBJECT_REF(Int32Generator, int32_t);
DECLARE_GENERATOR_OBJECT_REF(Int64Generator, int64_t);
DECLARE_GENERATOR_OBJECT_REF(Float32Generator, float);
DECLARE_GENERATOR_OBJECT_REF(Float64Generator, double);
DECLARE_GENERATOR_OBJECT_REF(RTValueGenerator, RTValue);

#undef DECLARE_GENERATOR_OBJECT_REF
}  // namespace runtime
}  // namespace matxscript
