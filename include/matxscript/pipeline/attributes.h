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

#include <limits>

#include <matxscript/runtime/container.h>

namespace matxscript {
namespace runtime {

static string_view PREFIX_KEY("____prefix_path____");

static constexpr int NONE_DEVICE = INT16_MIN;

struct Attributes {
 public:
  bool HasAttr(string_view key) {
    auto itr = attrs_.find(key);
    return itr != attrs_.end();
  }

  template <class U>
  U GetAttr(string_view key, const U& default_val = U{}) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    GenericValueConverter<U_TYPE> Converter;
    auto itr = attrs_.find(key);
    if (itr != attrs_.end()) {
      return Converter(itr->second);
    } else {
      return default_val;
    }
  }

  template <class U>
  void SetAttr(string_view key, U&& val) {
    using Converter = GenericValueConverter<RTValue>;
    attrs_[String(key.data(), key.size())] = Converter()(std::forward<U>(val));
  }

  static Attributes FromDict(const Dict& generic_attrs) {
    Attributes res_attrs;
    for (auto& attr : generic_attrs.items()) {
      MXCHECK(attr.first.type_code() == TypeIndex::kRuntimeString ||
              attr.first.type_code() == TypeIndex::kRuntimeUnicode);
      if (attr.first.type_code() == TypeIndex::kRuntimeString) {
        res_attrs.attrs_[attr.first.As<String>()] = attr.second;
      } else {
        res_attrs.attrs_[attr.first.As<Unicode>().encode()] = attr.second;
      }
    }
    return res_attrs;
  }

  Dict ToDict() const {
    Dict d;
    d.reserve(attrs_.size());
    for (auto& attr : attrs_) {
      d[attr.first] = attr.second;
    }
    return d;
  }

 protected:
  bool profiling = false;
  ska::flat_hash_map<String, RTValue> attrs_;

  friend class OpKernel;
  friend class TXSession;
};

template <>
inline String Attributes::GetAttr(string_view key, const String& default_val) const {
  auto itr = attrs_.find(key);
  if (itr != attrs_.end()) {
    if (itr->second.type_code() == TypeIndex::kRuntimeUnicode) {
      return UnicodeHelper::Encode(itr->second.AsNoCheck<unicode_view>());
    }
    return itr->second.As<String>();
  } else {
    return default_val;
  }
}

template <>
inline Unicode Attributes::GetAttr(string_view key, const Unicode& default_val) const {
  auto itr = attrs_.find(key);
  if (itr != attrs_.end()) {
    if (itr->second.type_code() == TypeIndex::kRuntimeString) {
      return StringHelper::Decode(itr->second.AsNoCheck<string_view>());
    }
    return itr->second.As<Unicode>();
  } else {
    return default_val;
  }
}

}  // namespace runtime
}  // namespace matxscript
