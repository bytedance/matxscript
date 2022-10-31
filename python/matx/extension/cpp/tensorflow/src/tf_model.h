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

// clang-format off
// !!! NOTICE !!!
// tensorflow header file has self-contained issue.
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
// clang-format on

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/ndarray.h>

namespace matxscript {
namespace runtime {

class TFEngine;
using TFEnginePtr = std::shared_ptr<TFEngine>;

class TFEngine {
 public:
  struct Options {
    String location;
    int device = -1;
    int use_xla = 0;
    bool allow_growth = true;
  };
  TFEngine() : tf_session_(nullptr), options_(){};
  virtual ~TFEngine() = default;

  void init(Options opt);

  Dict forward(const std::vector<std::pair<std::string, NDArray>>& inputs);

 private:
  static std::unique_ptr<tensorflow::Session> load_graph(const Options& opt,
                                                         tensorflow::GraphDef* graph_def);

  static std::unique_ptr<tensorflow::Session> load_saved_model(const Options& opt);

  static void read_signatures(const Options& opt,
                              std::vector<std::string>* inputs,
                              std::vector<std::string>* inputs_k,
                              std::vector<std::string>* outputs,
                              std::vector<std::string>* outputs_k);

 private:
  std::unique_ptr<tensorflow::Session> tf_session_;
  Options options_;
  std::vector<std::string> input_signatures_;
  std::vector<std::string> input_signatures_k_;
  std::unordered_map<std::string, std::string> input_signatures_mapping_;
  std::vector<std::string> output_signatures_;
  std::vector<std::string> output_signatures_k_;
};

class TFModel : public OpKernel {
 public:
  void Init() override;
  int Bundle(string_view folder) override;

  TFEnginePtr RegisterOrGetEngine(int device);

 private:
  std::mutex mutex_;
  String location_;
  int device_ = -1;
  int use_xla_ = 0;
  bool allow_growth_ = true;
  ska::flat_hash_map<int, TFEnginePtr> engines_;
};

class TensorProtoSerializer {
 public:
  String Serialize(const Any& val);
  RTValue Deserialize(const string_view& str);
};

}  // namespace runtime
}  // namespace matxscript
