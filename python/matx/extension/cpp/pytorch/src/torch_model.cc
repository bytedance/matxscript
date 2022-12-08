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
  torch::NoGradGuard no_grad;
  output = module_.forward(inputs);
}

int TorchModel::Bundle(string_view folder) {
  auto new_loc = BundlePath(location, folder);
  std::string example_file = folder.operator std::string() + "/" +
                             std::string(new_loc.data(), new_loc.size()) + ".example.json";
  if (!example.is_nullptr()) {
    auto example_json = pickle::Serialize(example);
    std::ofstream fc(example_file);
    MXCHECK(!fc.fail()) << "save " << example << " failed!";
    fc << example_json.view();
    fc.close();
  }
  SetAttr("location", std::move(new_loc));
  return 0;
}

TorchEnginePtr TorchModel::RegisterOrGetEngine(int device) {
  char key[4096] = {0};
  snprintf(key, 4096, "%d", device);
  std::lock_guard<std::mutex> lock(mutex_);
  auto itr = engines_.find(key);
  if (itr != engines_.end()) {
    return itr->second;
  } else {
    torch::NoGradGuard no_grad;
    auto e = std::make_shared<TorchEngine>();
    e->init(resource_path_ + location, device);
    engines_.emplace(key, e);
    return e;
  }
}

}  // namespace runtime
}  // namespace matxscript