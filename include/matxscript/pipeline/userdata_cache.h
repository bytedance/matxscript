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

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/json_util.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/singleton.h>

namespace matxscript {
namespace runtime {

class UserDataCache : public Singleton<UserDataCache> {
  typedef ska::flat_hash_map<String, UserDataRef> InstanceNameMap;
  typedef ska::flat_hash_map<String, InstanceNameMap> ClassGroupMap;
  typedef ska::flat_hash_map<String, ClassGroupMap> ScopeGroupMap;

 public:
  void AddRef(string_view scope) {
    std::lock_guard<std::mutex> lock(s_mutex_);
    {
      auto itr = scope_group_.find(scope);
      if (itr == scope_group_.end()) {
        scope_group_.emplace(scope, ClassGroupMap());
      }
    }
    {
      auto itr = scope_ref_.find(scope);
      if (itr == scope_ref_.end()) {
        scope_ref_.emplace(scope, 1);
      } else {
        ++(itr->second);
      }
    }
  }

  void DeRef(string_view scope) {
    std::lock_guard<std::mutex> lock(s_mutex_);
    auto itr = scope_ref_.find(scope);
    if (itr == scope_ref_.end()) {
      scope_group_.erase(scope);
    } else {
      --(itr->second);
      if (itr->second <= 0) {
        scope_group_.erase(scope);
        scope_ref_.erase(scope);
      }
    }
  }

  void Remove(string_view scope) {
    std::lock_guard<std::mutex> lock(s_mutex_);
    auto itr = scope_group_.find(scope);
    if (itr != scope_group_.end()) {
      scope_group_.erase(scope);
    }
    auto itr_ref = scope_ref_.find(scope);
    if (itr_ref != scope_ref_.end()) {
      scope_ref_.erase(scope);
    }
  }

  void Remove(string_view scope, string_view cls, string_view name) {
    std::lock_guard<std::mutex> lock(s_mutex_);
    auto itr = scope_group_.find(scope);
    if (itr == scope_group_.end()) {
      return;
    }
    auto itr2 = itr->second.find(cls);
    if (itr2 == itr->second.end()) {
      return;
    }
    itr2->second.erase(name);
  }

  UserDataRef Set(string_view scope, string_view cls, string_view name, UserDataRef p) {
    auto& scope_map = scope_group_;
    {
      auto itr = scope_map.find(scope);
      if (itr == scope_map.end()) {
        std::lock_guard<std::mutex> lock(s_mutex_);
        itr = scope_map.find(scope);
        if (itr == scope_map.end()) {
          scope_map.emplace(scope, ClassGroupMap());
        }
      }
    }
    auto& cls_map = scope_map.at(scope);
    {
      auto itr = cls_map.find(cls);
      if (itr == cls_map.end()) {
        std::lock_guard<std::mutex> lock(s_mutex_);
        itr = cls_map.find(cls);
        if (itr == cls_map.end()) {
          cls_map.emplace(cls, InstanceNameMap());
        }
      }
    }
    {
      auto& ptr_map = cls_map.at(cls);
      auto itr = ptr_map.find(name);
      if (itr == ptr_map.end()) {
        std::lock_guard<std::mutex> lock(s_mutex_);
        itr = ptr_map.find(name);
        if (itr == ptr_map.end()) {
          ptr_map.emplace(name, p);
          return p;
        } else {
          return itr->second;
        }
      } else {
        return itr->second;
      }
    }
  }

  UserDataRef Get(string_view scope, string_view cls, string_view name) {
    return __Get__(scope, cls, name);
  }

  const UserDataRef Get(string_view scope, string_view cls, string_view name) const {
    return __Get__(scope, cls, name);
  }

  UserDataRef __Get__(string_view scope, string_view cls, string_view name) const {
    auto& scope_map = scope_group_;
    {
      auto itr = scope_map.find(scope);
      if (itr == scope_map.end()) {
        return UserDataRef(nullptr);
      }
    }
    auto& cls_map = scope_map.at(scope);
    {
      auto itr = cls_map.find(cls);
      if (itr == cls_map.end()) {
        return UserDataRef(nullptr);
      }
    }
    {
      auto& ptr_map = cls_map.at(cls);
      auto itr = ptr_map.find(name);
      if (itr == ptr_map.end()) {
        return UserDataRef(nullptr);
      } else {
        return itr->second;
      }
    }
  }

  std::vector<UserDataRef> GetAll(string_view scope) {
    std::vector<UserDataRef> uds;
    auto& scope_map = scope_group_;
    auto itr = scope_map.find(scope);
    if (itr != scope_map.end()) {
      for (auto& cls_to_ins : itr->second) {
        for (auto& name_to_ins : cls_to_ins.second) {
          uds.emplace_back(name_to_ins.second);
        }
      }
    }
    return uds;
  }

 private:
  explicit UserDataCache() = default;
  ~UserDataCache() = default;
  friend class Singleton<UserDataCache>;
  DISALLOW_COPY_AND_ASSIGN(UserDataCache);

 private:
  ScopeGroupMap scope_group_;
  std::mutex s_mutex_;
  ska::flat_hash_map<String, int32_t> scope_ref_;
};

}  // namespace runtime
}  // namespace matxscript
