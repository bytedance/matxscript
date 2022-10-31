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

#include <matxscript/runtime/env_time.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

struct TimeLine {
  uint64_t stamp_start = 0;
  uint64_t stamp_end = 0;
};

class ProfilingHelper {
 public:
  explicit ProfilingHelper(TimeLine* tl = nullptr) {
    time_line_ = tl;
    if (time_line_) {
      time_line_->stamp_start = EnvTime::Default()->NowNanos();
    }
  }

  ~ProfilingHelper() {
    if (time_line_) {
      time_line_->stamp_end = EnvTime::Default()->NowNanos();
    }
  }

 private:
  TimeLine* time_line_;
};

}  // namespace runtime
}  // namespace matxscript
