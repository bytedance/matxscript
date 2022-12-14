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

#include <stdint.h>

namespace matxscript {
namespace runtime {

inline constexpr int64_t slice_index_correction(int64_t slice, int64_t len) noexcept {
  // change slice to [0, len)
  return slice < 0 ? (slice < -len ? 0 : slice + len) : (slice > len ? len : slice);
}

inline constexpr int64_t index_correction(int64_t index, int64_t len) noexcept {
  return index < 0 ? index + len : index;
}

}  // namespace runtime
}  // namespace matxscript
