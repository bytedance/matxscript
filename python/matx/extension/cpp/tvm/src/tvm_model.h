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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/container/ndarray.h>

namespace matxscript {
namespace runtime {

class TVMEngine;
using TVMEnginePtr = std::shared_ptr<TVMEngine>;

class TVMEngine {
 public:
  struct Options {
    String location;
    int device = -1;
    std::vector<String> output_names;
  };
  TVMEngine() : module_(), num_outputs_(0), ctx_(), options_(), lock_(){};
  virtual ~TVMEngine() = default;

  void init(Options opt);

  void forward(const std::vector<std::pair<std::string, const DLTensor*>>& tvm_inputs,
               std::vector<std::pair<std::string, ::tvm::runtime::NDArray>>* tvm_outputs);

 private:
  tvm::runtime::Module dso_module_;
  tvm::runtime::Module module_;
  int num_outputs_;
  TVMContext ctx_;
  Options options_;
  std::mutex lock_;
  tvm::runtime::PackedFunc func_get_input_;
  tvm::runtime::PackedFunc func_run_;
  tvm::runtime::PackedFunc func_get_output_;
};

class TVMModel : public OpKernel {
 public:
  void Init() override;
  int Bundle(string_view folder) override;
  TVMEnginePtr RegisterOrGetEngine(int device, bool share_model = true);

 private:
  std::mutex mutex_;
  String location_;
  std::vector<String> output_names_;
  ska::flat_hash_map<int, TVMEnginePtr> engines_;
};

}  // namespace runtime
}  // namespace matxscript
