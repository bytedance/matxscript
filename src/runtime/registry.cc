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
 * \file registry.cc
 * \brief The global registry of packed function.
 */
#include <matxscript/runtime/registry.h>
#include "runtime_base.h"

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

struct FunctionRegistry::Manager {
  ska::flat_hash_map<string_view, FunctionRegistry*> fmap;
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

FunctionRegistry& FunctionRegistry::Register(string_view name,
                                             bool can_override) {  // NOLINT(*)
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  if (m->fmap.count(name)) {
    MXCHECK(can_override) << "Global Function " << name << " is already registered";
  }

  FunctionRegistry* r = new FunctionRegistry();
  m->fmap[name] = r;
  return *r;
}

bool FunctionRegistry::Remove(string_view name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end())
    return false;
  m->fmap.erase(it);
  return true;
}

NativeFunction* FunctionRegistry::Get(string_view name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) {
    return nullptr;
  }
  return &(it->second->function_);
}

FunctionRegistry* FunctionRegistry::GetRegistry(string_view name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) {
    return nullptr;
  }
  return it->second;
}

std::vector<string_view> FunctionRegistry::ListNames() {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  std::vector<string_view> keys;
  keys.reserve(m->fmap.size());
  for (const auto& kv : m->fmap) {
    keys.push_back(kv.first);
  }
  return keys;
}

}  // namespace runtime
}  // namespace matxscript
