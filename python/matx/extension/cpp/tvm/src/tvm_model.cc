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
#include "tvm_model.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>

//#include <matxscript/runtime/buffered_file.h>
#include <matxscript/runtime/container/unicode_helper.h>
#include <matxscript/runtime/file_util.h>
#include <ostream>

namespace matxscript {
namespace runtime {

static std::string TVM_MODEL_STRUCT[4] = {
    "deploy_graph.json",
    "deploy_param.params",
    "deploy_lib.so",
    "shape_dict.params",
};

MATX_REGISTER_NATIVE_OP(TVMModel);

void TVMEngine::init(Options opt) {
  options_ = std::move(opt);
  if (options_.device < 0) {
    ctx_.device_id = 0;
    ctx_.device_type = kDLCPU;
  } else {
    ctx_.device_id = options_.device;
    ctx_.device_type = kDLGPU;
  }

  std::string graph_path = options_.location + "/" + TVM_MODEL_STRUCT[0];
  std::string params_path = options_.location + "/" + TVM_MODEL_STRUCT[1];
  std::string lib_path = options_.location + "/" + TVM_MODEL_STRUCT[2];
  if (!FileUtil::Exists(params_path) && !FileUtil::Exists(graph_path)) {
    // for TVM 0.8 and later...
    dso_module_ = ::tvm::runtime::Module::LoadFromFile(lib_path);
    module_ = dso_module_->GetFunction("default")(ctx_);
  } else {
    MXTHROW << "TVM model not found or incomplete at " << graph_path;
  }

  // INFO("load tvm module input data...");
  num_outputs_ = module_.GetFunction("get_num_outputs")();
  MXCHECK_EQ(options_.output_names.size(), num_outputs_);

  func_run_ = module_.GetFunction("run");
  MXCHECK(func_run_ != nullptr) << "[TVMEngine] run function is not found";
  func_get_output_ = module_.GetFunction("get_output");
  MXCHECK(func_get_output_ != nullptr) << "[TVMEngine] get_output function is not found";
  func_get_input_ = module_.GetFunction("get_input");
  MXCHECK(func_get_input_ != nullptr) << "[TVMEngine] get_input function is not found";

  MXLOG(INFO) << "[TVMEngine] finish init";
}

void TVMEngine::forward(const std::vector<std::pair<std::string, const DLTensor*>>& tvm_inputs,
                        std::vector<std::pair<std::string, ::tvm::runtime::NDArray>>* tvm_outputs) {
  // cpu mode remove lock ?
  std::lock_guard<std::mutex> guard(lock_);
  for (size_t i = 0; i < tvm_inputs.size(); ++i) {
    ::tvm::runtime::NDArray input_tensor = func_get_input_(tvm_inputs[i].first);
    const DLTensor* dl_tsr = tvm_inputs[i].second;
    input_tensor.CopyFromBytes(reinterpret_cast<char*>(dl_tsr->data) + dl_tsr->byte_offset,
                               GetDataSize(*dl_tsr));
  }
  func_run_();
  for (int i = 0; i < num_outputs_; ++i) {
    const auto& name_o = options_.output_names[i];
    ::tvm::runtime::NDArray tvm_arr = func_get_output_(i);
    tvm_outputs->emplace_back(name_o, std::move(tvm_arr));
  }
}

void TVMModel::Init() {
  location_ = GetAttr<Unicode>("location").encode();
  List outputs = GetAttr<List>("outputs");
  output_names_.clear();
  for (auto& item : outputs) {
    String s = item.As<Unicode>().encode();
    output_names_.push_back(s);
  }
}

int TVMModel::Bundle(string_view folder) {
  auto new_loc = BundlePath(location_, folder);
  SetAttr("location", new_loc.decode());
  return 0;
}

TVMEnginePtr TVMModel::RegisterOrGetEngine(int device, bool share_model) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (share_model) {
    auto itr = engines_.find(device);
    if (itr != engines_.end()) {
      return itr->second;
    } else {
      auto e = std::make_shared<TVMEngine>();
      TVMEngine::Options opt;
      opt.output_names = output_names_;
      opt.location = resource_path_ + location_;
      opt.device = device;
      e->init(opt);
      engines_.emplace(device, e);
      return e;
    }
  } else {
    auto e = std::make_shared<TVMEngine>();
    TVMEngine::Options opt;
    opt.output_names = output_names_;
    opt.location = resource_path_ + location_;
    opt.device = device;
    e->init(opt);
    return e;
  }
}

}  // namespace runtime
}  // namespace matxscript
