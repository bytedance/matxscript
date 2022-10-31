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
#include <matxscript/runtime/native_object_registry.h>

#include <mutex>

#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/generic/generic_funcs.h>
#include <matxscript/runtime/native_object_maker.h>

namespace matxscript {
namespace runtime {

struct NativeObjectRegistry::Manager {
  ska::flat_hash_map<string_view, NativeObjectRegistry*> fmap;
  // mutex
  std::mutex mutex;
  Manager() = default;
  static Manager* Global() {
    // We deliberately leak the Manager instance, to avoid leak sanitizers
    // complaining about the entries in Manager::fmap being leaked at program
    // exit.
    static Manager* inst = new Manager();
    return inst;
  }
};

NativeObjectRegistry& NativeObjectRegistry::Register(string_view name,
                                                     bool can_override) {  // NOLINT(*)
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  if (m->fmap.count(name)) {
    MXCHECK(can_override) << "Global Class " << name << " is already registered";
  }

  NativeObjectRegistry* r = new NativeObjectRegistry();
  m->fmap[name] = r;
  return *r;
}

bool NativeObjectRegistry::Remove(string_view name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end())
    return false;
  m->fmap.erase(it);
  return true;
}

NativeObjectRegistry* NativeObjectRegistry::Get(string_view name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) {
    return nullptr;
  }
  return (it->second);
}

std::vector<string_view> NativeObjectRegistry::ListNames() {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  std::vector<string_view> keys;
  keys.reserve(m->fmap.size());
  for (const auto& kv : m->fmap) {
    keys.push_back(kv.first);
  }
  return keys;
}

std::vector<string_view> NativeObjectRegistry::ListPureObjNames() {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  std::vector<string_view> keys;
  for (const auto& kv : m->fmap) {
    if (!kv.second->is_native_op_ && !kv.second->is_jit_object_) {
      keys.push_back(kv.first);
    }
  }
  return keys;
}

}  // namespace runtime
}  // namespace matxscript
