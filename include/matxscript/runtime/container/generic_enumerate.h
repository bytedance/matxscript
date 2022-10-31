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

#include <type_traits>

namespace matxscript {
namespace runtime {

namespace details {
template <class Iterable>
struct generic_enumerate : private Iterable {
  int64_t pos = 0;

  generic_enumerate() {
  }
  generic_enumerate(Iterable seq, int64_t first) : Iterable(seq), pos(first) {
  }
  bool HasNext() const {
    return Iterable::HasNext();
  }
  auto Next() {
    return std::make_pair(pos++, Iterable::Next());
  }
  auto Next(bool* has_next) {
    return std::make_pair(pos++, Iterable::Next(has_next));
  }
};

}  // namespace details

template <class Iterable>
details::generic_enumerate<
    typename std::remove_cv<typename std::remove_reference<Iterable>::type>::type>
generic_enumerate(Iterable&& seq, long first = 0L) {
  return {std::forward<Iterable>(seq), first};
}

}  // namespace runtime
}  // namespace matxscript
