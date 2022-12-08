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

#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/script.h>

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/pipeline/pickle.h>
#include <matxscript/runtime/file_util.h>

namespace matxscript {
namespace runtime {

class TorchEngine;
using TorchEnginePtr = std::shared_ptr<TorchEngine>;

class TorchEngine {
 public:
  TorchEngine() : torch_device_(at::kCPU), module_(){};
  virtual ~TorchEngine() = default;

  void init(const std::string& path, int device) {
    torch::NoGradGuard no_grad;
    MXCHECK(FileUtil::Exists(path)) << "[TorchEngine] model location not exist: " << path;
    if (device >= 0) {
      torch_device_ = at::Device(at::kCUDA, device);
    } else {
      torch_device_ = at::Device(at::kCPU);
    }
    try {
      module_ = torch::jit::load(path, torch_device_);
    } catch (const std::exception& e) {
      MXTHROW << "[TorchEngine] Load PyTorch model failed: " << e.what();
    }
  }

  void forward(const std::vector<torch::jit::IValue>& inputs, torch::jit::IValue& output);

  at::Device get_device() const {
    return torch_device_;
  }

 private:
  at::Device torch_device_;
  torch::jit::script::Module module_;
};

class TorchModel : public OpKernel {
 public:
  void Init() override {
    location = GetAttr<Unicode>("location").encode();
    if (HasAttr("example")) {
      example = GetAttr<RTValue>("example");
    }
  }
  int Bundle(string_view folder) override;
  TorchEnginePtr RegisterOrGetEngine(int device);

 private:
  String location;
  RTValue example;
  std::mutex mutex_;
  std::unordered_map<std::string, TorchEnginePtr> engines_;
  friend class TorchInferOp;
};

}  // namespace runtime
}  // namespace matxscript
