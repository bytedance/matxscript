// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
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

/*!
 * \file matx/runtime/attr_registry.h
 * \brief Common global registry for objects that also have additional attrs.
 */
#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <matxscript/ir/_base/attr_registry_map.h>
#include <matxscript/ir/_base/cow_array_ref.h>
#include <matxscript/ir/_base/cow_map_ref.h>
#include <matxscript/runtime/container.h>

namespace matxscript {
namespace runtime {

/*!
 * \brief Implementation of registry with attributes.
 *
 * \tparam EntryType The type of the registry entry.
 * \tparam KeyType The actual key that is used to lookup the attributes.
 *                 each entry has a corresponding key by default.
 */
template <typename EntryType, typename KeyType>
class AttrRegistry {
 public:
  using TSelf = AttrRegistry<EntryType, KeyType>;
  /*!
   * \brief Get an entry from the registry.
   * \param name The name of the item.
   * \return The corresponding entry.
   */
  const EntryType* Get(const StringRef& name) const {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end())
      return it->second;
    return nullptr;
  }

  /*!
   * \brief Get an entry or register a new one.
   * \param name The name of the item.
   * \return The corresponding entry.
   */
  EntryType& RegisterOrGet(const StringRef& name) {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end())
      return *it->second;
    uint32_t registry_index = static_cast<uint32_t>(entries_.size());
    auto entry = std::unique_ptr<EntryType>(new EntryType(registry_index));
    auto* eptr = entry.get();
    eptr->name = name.operator String();
    entry_map_[name] = eptr;
    entries_.emplace_back(std::move(entry));
    return *eptr;
  }

  /*!
   * \brief List all the entry names in the registry.
   * \return The entry names.
   */
  Array<StringRef> ListAllNames() const {
    Array<StringRef> names;
    for (const auto& kv : entry_map_) {
      names.push_back(kv.first);
    }
    return names;
  }

  /*!
   * \brief Update the attribute stable.
   * \param attr_name The name of the attribute.
   * \param key The key to the attribute table.
   * \param value The value to be set.
   * \param plevel The support level.
   */
  void UpdateAttr(const StringRef& attr_name, const KeyType& key, RTValue value, int plevel) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& op_map = attrs_[attr_name];
    if (op_map == nullptr) {
      op_map.reset(new AttrRegistryMapContainerMap<KeyType>());
      op_map->attr_name_ = attr_name;
    }

    uint32_t index = key->AttrRegistryIndex();
    if (op_map->data_.size() <= index) {
      op_map->data_.resize(index + 1, std::make_pair(RTValue(), 0));
    }
    std::pair<RTValue, int>& p = op_map->data_[index];
    MXCHECK(p.second != plevel) << "Attribute " << attr_name << " of " << key->AttrRegistryName()
                                << " is already registered with same plevel=" << plevel;
    MXCHECK(value.type_code() != runtime::TypeIndex::kRuntimeNullptr)
        << "Registered packed_func is Null for " << attr_name << " of operator "
        << key->AttrRegistryName();
    if (p.second < plevel && value.type_code() != runtime::TypeIndex::kRuntimeNullptr) {
      op_map->data_[index] = std::make_pair(value, plevel);
    }
  }

  /*!
   * \brief Reset an attribute table entry.
   * \param attr_name The name of the attribute.
   * \param key The key to the attribute table.
   */
  void ResetAttr(const StringRef& attr_name, const KeyType& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& op_map = attrs_[attr_name];
    if (op_map == nullptr) {
      return;
    }
    uint32_t index = key->AttrRegistryIndex();
    if (op_map->data_.size() > index) {
      op_map->data_[index] = std::make_pair(RTValue(), 0);
    }
  }

  /*!
   * \brief Get an internal attribute map.
   * \param attr_name The name of the attribute.
   * \return The result attribute map.
   */
  const AttrRegistryMapContainerMap<KeyType>& GetAttrMap(const StringRef& attr_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) {
      MXLOG(FATAL) << "Attribute \'" << attr_name << "\' is not registered";
    }
    return *it->second.get();
  }

  /*!
   * \brief Check of attribute has been registered.
   * \param attr_name The name of the attribute.
   * \return The check result.
   */
  bool HasAttrMap(const StringRef& attr_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return attrs_.count(attr_name);
  }

  /*!
   * \return a global singleton of the registry.
   */
  static TSelf* Global() {
    static TSelf* inst = new TSelf();
    return inst;
  }

 private:
  // mutex to avoid registration from multiple threads.
  std::mutex mutex_;
  // entries in the registry
  std::vector<std::unique_ptr<EntryType>> entries_;
  // map from name to entries.
  std::unordered_map<StringRef, EntryType*> entry_map_;
  // storage of additional attribute table.
  std::unordered_map<StringRef, std::unique_ptr<AttrRegistryMapContainerMap<KeyType>>> attrs_;
};

}  // namespace runtime
}  // namespace matxscript
