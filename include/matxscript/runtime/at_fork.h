// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file is inspired by folly AtFork.
 * https://github.com/facebook/folly/blob/main/folly/system/AtFork.h
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#pragma once

#include <cstddef>
#include <cstdint>

#include <functional>
#include <list>
#include <map>
#include <mutex>

namespace matxscript {
namespace runtime {
namespace internal {

struct AtFork {
 public:
  static void RegisterHandler(void const* handle,
                              std::function<bool()> prepare,
                              std::function<void()> parent,
                              std::function<void()> child);

  static void UnregisterHandler(void const* handle);
};

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript
