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
#include "torch_model.h"

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(TorchModel);

void TorchEngine::forward(const std::vector<torch::jit::IValue>& inputs,
                          torch::jit::IValue& output) {
  output = module_.forward(inputs);
}

TorchEnginePtr TorchModel::RegisterOrGetEngine(int device) {
  char key[4096] = {0};
  snprintf(key, 4096, "%d", device);
  std::lock_guard<std::mutex> lock(mutex_);
  auto itr = engines_.find(key);
  if (itr != engines_.end()) {
    return itr->second;
  } else {
    auto e = std::make_shared<TorchEngine>();
    e->init(resource_path_ + location, device);
    engines_.emplace(key, e);
    return e;
  }
}

}  // namespace runtime
}  // namespace matxscript