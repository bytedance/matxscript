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

#include <functional>
#include <typeindex>
#include <typeinfo>

#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/itertor_ref.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

struct FTObjectBaseNode : public Object {
  using NativeMethod = std::function<RTValue(RTView self, PyArgs args)>;
  using FunctionTable = ska::flat_hash_map<string_view, NativeMethod>;

  FTObjectBaseNode(const FunctionTable* funcs, const std::type_index* info, uint64_t tag)
      : child_function_table_(funcs), child_type_index_(info), tag(tag) {
  }
  FTObjectBaseNode(const FTObjectBaseNode&) = default;
  FTObjectBaseNode(FTObjectBaseNode&&) = default;
  FTObjectBaseNode& operator=(const FTObjectBaseNode&) = default;
  FTObjectBaseNode& operator=(FTObjectBaseNode&&) = default;

  // function table is unbound
  const FunctionTable* child_function_table_ = nullptr;
  const std::type_index* child_type_index_ = nullptr;
  uint64_t tag = 0;

  static FTObjectBaseNode* GetMutableSelfPtr(const RTValue& o) {
    return static_cast<FTObjectBaseNode*>(const_cast<RTValue&>(o).ptr<Object>());
  }

  static uint32_t _RegisterRuntimeTypeIndex(string_view key,
                                            uint32_t static_tindex,
                                            uint32_t parent_tindex,
                                            uint32_t num_child_slots,
                                            bool child_slots_can_overflow);

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeFTObjectBase;
  static constexpr const char* _type_key = "FTContainerBase";
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(FTObjectBaseNode, Object);
};

struct FTObjectBase : public ObjectRef {
  using ContainerType = FTObjectBaseNode;

  FTObjectBase() noexcept = default;
  explicit FTObjectBase(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }
  FTObjectBase(const FTObjectBase& other) noexcept = default;
  FTObjectBase(FTObjectBase&& other) noexcept = default;
  FTObjectBase& operator=(const FTObjectBase& other) noexcept = default;
  FTObjectBase& operator=(FTObjectBase&& other) noexcept = default;

  const FTObjectBaseNode* operator->() const {
    return static_cast<const FTObjectBaseNode*>(data_.get());
  }
  const FTObjectBaseNode* get() const {
    return operator->();
  }

  RTValue generic_call_attr(string_view func_name, PyArgs args) const;
};

std::ostream& operator<<(std::ostream& os, FTObjectBase const& n);

inline std::type_index GetFTObjectBaseStdTypeIndex(const Object* o) noexcept {
  return *(static_cast<const FTObjectBaseNode*>(o)->child_type_index_);
}

template <class T1, class T2>
struct root_type_is_same {
  using TYPE1 = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
  using TYPE2 = typename std::remove_cv<typename std::remove_reference<T2>::type>::type;
  static constexpr bool value =
      TypeIndex::type_index_traits<TYPE1>::value != TypeIndex::kRuntimeUnknown &&
      TypeIndex::type_index_traits<TYPE1>::value == TypeIndex::type_index_traits<TYPE2>::value;
};

template <class T1, class T2>
struct root_type_is_convertible {
  using TYPE1 = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
  using TYPE2 = typename std::remove_cv<typename std::remove_reference<T2>::type>::type;
  static constexpr bool value =
      root_type_is_same<TYPE1, TYPE2>::value ||
      (TypeIndex::type_index_traits<TYPE1>::value == TypeIndex::kRuntimeFTList &&
       TypeIndex::type_index_traits<TYPE2>::value == TypeIndex::kRuntimeList) ||
      (TypeIndex::type_index_traits<TYPE1>::value == TypeIndex::kRuntimeList &&
       TypeIndex::type_index_traits<TYPE2>::value == TypeIndex::kRuntimeFTList) ||
      (TypeIndex::type_index_traits<TYPE1>::value == TypeIndex::kRuntimeFTSet &&
       TypeIndex::type_index_traits<TYPE2>::value == TypeIndex::kRuntimeSet) ||
      (TypeIndex::type_index_traits<TYPE1>::value == TypeIndex::kRuntimeSet &&
       TypeIndex::type_index_traits<TYPE2>::value == TypeIndex::kRuntimeFTSet) ||
      (TypeIndex::type_index_traits<TYPE1>::value == TypeIndex::kRuntimeFTDict &&
       TypeIndex::type_index_traits<TYPE2>::value == TypeIndex::kRuntimeDict) ||
      (TypeIndex::type_index_traits<TYPE1>::value == TypeIndex::kRuntimeDict &&
       TypeIndex::type_index_traits<TYPE2>::value == TypeIndex::kRuntimeFTDict);
};

namespace type_relation {
using same_root = std::integral_constant<int, 0>;
using one_is_any = std::integral_constant<int, 1>;
using root_is_convertible = std::integral_constant<int, 2>;
using no_rel = std::false_type;

template <class T1, class T2>
struct traits {
  using TYPE1 = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
  using TYPE2 = typename std::remove_cv<typename std::remove_reference<T2>::type>::type;
  using type = typename std::conditional<
      std::is_same<TYPE1, TYPE2>::value || root_type_is_same<TYPE1, TYPE2>::value,
      same_root,
      typename std::conditional<
          std::is_same<RTValue, TYPE1>::value || std::is_same<RTValue, TYPE2>::value,
          one_is_any,
          typename std::conditional<root_type_is_convertible<TYPE1, TYPE2>::value,
                                    root_is_convertible,
                                    no_rel>::type>::type>::type;
};
}  // namespace type_relation

}  // namespace runtime
}  // namespace matxscript
