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
#include <matxscript/runtime/env_time.h>

#include <ctime>

namespace matxscript {
namespace runtime {

class PosixEnvTime : public EnvTime {
 public:
  PosixEnvTime() = default;

  uint64_t NowNanos() const override {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos + static_cast<uint64_t>(ts.tv_nsec));
  }
};

EnvTime* EnvTime::Default() {
  static EnvTime* default_env_time = new PosixEnvTime;
  return default_env_time;
}

}  // namespace runtime
}  // namespace matxscript
