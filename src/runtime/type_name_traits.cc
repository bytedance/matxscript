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
#include <matxscript/runtime/type_name_traits.h>

#include <mutex>

namespace matxscript {
namespace runtime {

struct TypeNameTraits::Manager {
  // mutex
  static std::mutex s_mutex_;
  static TypeNameTraits* s_instance_;
  Manager() = default;
  static TypeNameTraits* Global() {
    if (s_instance_ == nullptr) {
      std::lock_guard<std::mutex> lock(s_mutex_);
      if (s_instance_ == nullptr) {
        s_instance_ = new TypeNameTraits();
      }
    }
    return s_instance_;
  }
};
std::mutex TypeNameTraits::Manager::s_mutex_{};
TypeNameTraits* TypeNameTraits::Manager::s_instance_{nullptr};

TypeNameTraits& TypeNameTraits::Register(std::type_index ty_idx, string_view name) {
  Manager::Global()->type_info_[ty_idx] = name;
  return *Manager::Global();
}

string_view TypeNameTraits::Get(std::type_index ty_idx) {
  auto ptr = Manager::Global();
  auto itr = ptr->type_info_.find(ty_idx);
  return itr == ptr->type_info_.end() ? "" : itr->second;
}

}  // namespace runtime
}  // namespace matxscript
