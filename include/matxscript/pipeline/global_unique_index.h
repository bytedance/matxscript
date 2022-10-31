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

#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/runtime_port.h>
#include <matxscript/runtime/singleton.h>

namespace matxscript {
namespace runtime {

class GlobalUniqueIndex : public Singleton<GlobalUniqueIndex> {
  friend class Singleton<GlobalUniqueIndex>;

 public:
  uint64_t gen_index(string_view name = "__global__") {
    std::lock_guard<std::mutex> lock(mutex_base_);
    auto itr = index_info_.find(name);
    if (itr == index_info_.end()) {
      index_info_.emplace(std::string(name.data(), name.size()), 0);
      itr = index_info_.find(name);
    }
    return itr->second++;
  }

  std::string gen_uniq_name(string_view prefix, string_view seed) {
    std::lock_guard<std::mutex> lock(mutex_app_);
    std::string result;
    result.reserve(prefix.size() + 24);
    result.append(prefix.data(), prefix.size());
    result.push_back('_');
    auto hash_value = std::hash<string_view>()(seed);
    result.append(std::to_string(hash_value));
    auto suffix = gen_index(result);
    result.push_back('_');
    result.append(std::to_string(suffix));
    return result;
  }

  uint64_t gen_uniq_signature(string_view seed) {
    std::lock_guard<std::mutex> lock(mutex_app_);
    uint64_t hash_value = BytesHash(seed.data(), seed.size());
    uint64_t result = hash_value;
    uint64_t offset = 1;
    while (signature_info_.count(result)) {
      result = hash_value + ((offset++) << 32);
    }
    signature_info_.emplace(result);
    return result;
  }

 private:
  ska::flat_hash_map<std::string, uint64_t, std_string_hash, std_string_equal_to> index_info_{};
  ska::flat_hash_set<uint64_t> signature_info_{};
  std::mutex mutex_base_;
  std::mutex mutex_app_;

 private:
  DISALLOW_COPY_AND_ASSIGN(GlobalUniqueIndex);
  GlobalUniqueIndex() = default;
  ~GlobalUniqueIndex() = default;
};

}  // namespace runtime
}  // namespace matxscript
