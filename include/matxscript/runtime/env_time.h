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

#include <cstdint>

namespace matxscript {
namespace runtime {

class EnvTime {
 public:
  static constexpr uint64_t kMicrosToPicos = 1000ULL * 1000ULL;
  static constexpr uint64_t kMicrosToNanos = 1000ULL;
  static constexpr uint64_t kMillisToMicros = 1000ULL;
  static constexpr uint64_t kMillisToNanos = 1000ULL * 1000ULL;
  static constexpr uint64_t kSecondsToMillis = 1000ULL;
  static constexpr uint64_t kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() = default;
  virtual ~EnvTime() = default;

  /// \brief Returns a default impl suitable for the current operating
  /// system.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static EnvTime* Default();

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  virtual uint64_t NowNanos() const = 0;

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64_t NowMicros() const {
    return NowNanos() / kMicrosToNanos;
  }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64_t NowSeconds() const {
    return NowNanos() / kSecondsToNanos;
  }
};

}  // namespace runtime
}  // namespace matxscript
