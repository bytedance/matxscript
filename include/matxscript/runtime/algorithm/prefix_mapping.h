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

#include <map>
#include <memory>
#include <set>

#include <matxscript/runtime/algorithm/cedar.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {

// Given a list of strings, finds the longest string which is a
// prefix of a query.
class PrefixMapping {
#if defined(USE_CEDAR_UNORDERED)
  typedef cedar::da<int, -1, -2, false> cedar_t;
#else
  typedef cedar::da<int> cedar_t;
#endif
 public:
  explicit PrefixMapping(const std::map<String, int>& dic);

  // Finds the longest string in dic, which is a prefix of `w`.
  // Returns the UTF8 byte length of matched string.
  // `found` is set if a prefix match exists.
  // If no entry is found, return 0.
  int PrefixSearch(const char* w, size_t w_len, int* val) const;

 private:
  std::unique_ptr<cedar_t> trie_;
};

}  // namespace runtime
}  // namespace matxscript
