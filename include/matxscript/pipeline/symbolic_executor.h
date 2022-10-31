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

#include <memory>

#include <matxscript/pipeline/node.h>
#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/_flat_hash_map.h>

namespace matxscript {
namespace runtime {

struct SymbolicExecutor {
  static std::vector<std::unique_ptr<Symbol>> Compose(
      OpKernelPtr op,
      const std::vector<const Symbol*>& args,
      const ska::flat_hash_map<String, const Symbol*>& kwargs,
      int num_output);

  static inline std::vector<std::unique_ptr<Symbol>> Compose(OpKernelPtr op,
                                                             const std::vector<const Symbol*>& args,
                                                             int num_output) {
    static ska::flat_hash_map<String, const Symbol*> kwargs{};
    return Compose(op, args, kwargs, num_output);
  }
};

}  // namespace runtime
}  // namespace matxscript
