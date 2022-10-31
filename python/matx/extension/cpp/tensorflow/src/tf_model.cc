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
#include "tf_model.h"

// clang-format off
// !!! NOTICE !!!
// tensorflow header file has self-contained issue.
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/version.h>

#if defined(MATX_SCRIPT_TENSORFLOW_PYTHON_MODE) && TF_MAJOR_VERSION == 1
#include "_tensorflow_cc_savedmodel_reader.h"
#include "_tensorflow_cc_savedmodel_constants.h"
#else
#include <tensorflow/cc/saved_model/reader.h>
#include <tensorflow/cc/saved_model/constants.h>
#endif

#ifdef MATX_SCRIPT_TENSORFLOW_PYTHON_MODE
#include "_tensorflow_cc_savedmodel_tag_constants.h"
#include "_tensorflow_cc_savedmodel_signature_constants.h"
#else
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/cc/saved_model/signature_constants.h>
#endif
// clang-format on

#include "tf_utils.h"

#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/env_time.h>
#include <matxscript/runtime/file_util.h>

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(TFModel);

void TFEngine::init(Options opt) {
  options_ = std::move(opt);
  read_signatures(options_,
                  &input_signatures_,
                  &input_signatures_k_,
                  &output_signatures_,
                  &output_signatures_k_);
  for (auto i = 0; i < input_signatures_.size(); ++i) {
    input_signatures_mapping_[input_signatures_k_[i]] = input_signatures_[i];
  }
  tf_session_ = load_saved_model(options_);
}

Dict TFEngine::forward(const std::vector<std::pair<std::string, NDArray>>& inputs) {
  std::vector<std::pair<std::string, tensorflow::Tensor>> tf_inputs;
  tf_inputs.reserve(inputs.size());
  for (auto& input_kv : inputs) {
    auto iter = input_signatures_mapping_.find(input_kv.first);
    if (iter != input_signatures_mapping_.end()) {
      tf_inputs.push_back(std::make_pair(iter->second, tf_utils::ToTFTensor(input_kv.second)));
    } else {
      tf_inputs.push_back(std::make_pair(input_kv.first, tf_utils::ToTFTensor(input_kv.second)));
    }
  }
  std::vector<tensorflow::Tensor> tf_outputs;

  // trace with run_meta
  // tensorflow::RunOptions options;
  // options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
  // tensorflow::RunMetadata run_meta;

  // auto stat = tf_session_->Run(
  //    options, tf_inputs, *out_sigs, {}, &tf_outputs, &run_meta);
  auto stat = tf_session_->Run(tf_inputs, output_signatures_, {}, &tf_outputs);
  MXCHECK(stat.ok()) << "tensorflow session run failed with error: " << stat;

  Dict result;
  result.reserve(output_signatures_.size());
  for (size_t i = 0; i < output_signatures_.size(); ++i) {
    result[StringHelper::Decode(string_view(output_signatures_k_[i]))] =
        tf_utils::FromTFTensor(tf_outputs[i]);
  }
  return result;
}

std::unique_ptr<tensorflow::Session> TFEngine::load_graph(const Options& opt,
                                                          tensorflow::GraphDef* graph_def) {
  auto jit_level = tensorflow::OptimizerOptions_GlobalJitLevel_DEFAULT;
  if (opt.use_xla == 1) {
    jit_level = tensorflow::OptimizerOptions_GlobalJitLevel_ON_1;
  } else if (opt.use_xla == 2) {
    jit_level = tensorflow::OptimizerOptions_GlobalJitLevel_ON_2;
  }

  if (opt.device >= 0) {
    tensorflow::graph::SetDefaultDevice("/device:GPU:" + std::to_string(opt.device), graph_def);
  }

  auto options = tensorflow::SessionOptions();
  // For some reason SetDefaultDevice sets /gpu:0 into visible_device_list
  // reset it to empty here.
  options.config.mutable_gpu_options()->set_visible_device_list("");
  options.config.set_allow_soft_placement(true);
  options.config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(
      jit_level);
  if (opt.device < 0) {
    (*options.config.mutable_device_count())["GPU"] = 0;
  } else {
    options.config.mutable_gpu_options()->set_allow_growth(opt.allow_growth);
  }
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
  tensorflow::Status cstat = session->Create(*graph_def);
  MXCHECK(cstat.ok()) << "Failed to create tensorflow session with error: " << cstat;
  return session;
}

std::unique_ptr<tensorflow::Session> TFEngine::load_saved_model(const Options& opt) {
  MXCHECK(FileUtil::Exists(opt.location))
      << "[TFEngine] savedmodel not found, location: " << opt.location;
  auto saved_model_pb = opt.location + "/" + "saved_model.pb";
  MXCHECK(FileUtil::Exists(saved_model_pb))
      << "[TFEngine] saved_model.pb not found, location: " << saved_model_pb;
  tensorflow::MetaGraphDef meta_graph_def;
  ReadMetaGraphDefFromSavedModel(opt.location, {tensorflow::kSavedModelTagServe}, &meta_graph_def);
  auto session = load_graph(opt, meta_graph_def.mutable_graph_def());

  const std::string variables_path = opt.location + "/" +
                                     tensorflow::kSavedModelVariablesDirectory + "/" +
                                     tensorflow::kSavedModelVariablesFilename;
  tensorflow::Tensor variables_path_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
#if TF_MAJOR_VERSION == 2
  variables_path_tensor.scalar<tensorflow::tstring>()() = variables_path;
#elif TF_MAJOR_VERSION == 1
  variables_path_tensor.scalar<std::string>()() = variables_path;
#endif
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {std::string(meta_graph_def.saver_def().filename_tensor_name()), variables_path_tensor}};
  std::vector<tensorflow::Tensor> tf_outputs;
  session->Run(inputs, {}, {meta_graph_def.saver_def().restore_op_name()}, &tf_outputs);
  return session;
}

void TFEngine::read_signatures(const Options& opt,
                               std::vector<std::string>* inputs,
                               std::vector<std::string>* inputs_k,
                               std::vector<std::string>* outputs,
                               std::vector<std::string>* outputs_k) {
  // INFO("[TFEngine] begin read_signatures");
  MXCHECK(FileUtil::Exists(opt.location))
      << "[TFEngine] savedmodel not found, location: " << opt.location;
  auto saved_model_pb = opt.location + "/" + "saved_model.pb";
  MXCHECK(FileUtil::Exists(saved_model_pb))
      << "[TFEngine] saved_model.pb not found, location: " << saved_model_pb;
  inputs->clear();
  inputs_k->clear();
  outputs->clear();
  outputs_k->clear();
  tensorflow::MetaGraphDef meta_graph_def;
  ReadMetaGraphDefFromSavedModel(opt.location, {tensorflow::kSavedModelTagServe}, &meta_graph_def);
  std::string signature_name = tensorflow::kDefaultServingSignatureDefKey;
  auto& signature_defs = meta_graph_def.signature_def();
  auto iter = signature_defs.find(signature_name);
  if (iter == signature_defs.end()) {
    MXCHECK(false) << "[TFEngine] no signature def: " << signature_name;
  }
  tensorflow::SignatureDef signature_def = iter->second;
  if (signature_def.inputs().empty()) {
    MXCHECK(false) << "[TFEngine] input signatures is empty";
  }
  if (signature_def.outputs().empty()) {
    MXCHECK(false) << "[TFEngine] output signatures is empty";
  }
  for (auto& input_iter : signature_def.inputs()) {
    // printf("[TFEngine] [%s] inputs: signature: %s\n", opt.location.c_str(),
    // input_iter.first.c_str());
    inputs_k->push_back(input_iter.first);
    inputs->push_back(input_iter.second.name());
  }
  for (auto& output_iter : signature_def.outputs()) {
    // printf("[TFEngine] [%s] outputs: signature: %s\n", opt.location.c_str(),
    // output_iter.first.c_str());
    outputs->push_back(output_iter.second.name());
    outputs_k->push_back(output_iter.first);
  }
  // INFO("[TFEngine] finish read_signatures");
}

void TFModel::Init() {
  location_ = GetAttr<Unicode>("location").encode();
  use_xla_ = GetAttr<int>("use_xla");
  allow_growth_ = GetAttr<bool>("allow_growth");
}

int TFModel::Bundle(string_view folder) {
  auto new_loc = BundlePath(location_, folder);
  SetAttr("location", new_loc.decode());

  return 0;
}

TFEnginePtr TFModel::RegisterOrGetEngine(int device) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto itr = engines_.find(device);
  if (itr != engines_.end()) {
    return itr->second;
  } else {
    auto e = std::make_shared<TFEngine>();
    TFEngine::Options opt;
    opt.location = resource_path_ + location_;
    opt.device = device;
    opt.use_xla = use_xla_;
    opt.allow_growth = allow_growth_;
    e->init(opt);
    engines_.emplace(device, e);
    return e;
  }
}

String TensorProtoSerializer::Serialize(const Any& val) {
  if (val.type_code() == TypeIndex::kRuntimeNDArray) {
    auto view = val.AsObjectViewNoCheck<NDArray>();
    auto tf_tsr = tf_utils::ToTFTensor(view.data());
    std::string serialize_data;
    tensorflow::TensorProto tensor_proto;
    tf_tsr.AsProtoField(&tensor_proto);
    tensor_proto.SerializeToString(&serialize_data);
    return String(serialize_data);
  } else {
    MXTHROW << "TensorProtoSerializer::Serialize : " << val.type_name() << " is not supported";
  }
}

RTValue TensorProtoSerializer::Deserialize(const string_view& str) {
  tensorflow::TensorProto tensor_proto;
  tensorflow::Tensor tensor;
  if (!tensor_proto.ParseFromArray(str.data(), str.size())) {
    MXTHROW << "TensorProtoSerializer::Deserialize : invalid input";
  }
  if (!tensor.FromProto(tensor_proto)) {
    MXTHROW << "TensorProtoSerializer::Deserialize : invalid input";
  }
  return tf_utils::FromTFTensor(tensor);
}

MATX_REGISTER_NATIVE_OBJECT(TensorProtoSerializer)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<TensorProtoSerializer>();
    })
    .RegisterFunction("serialize",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 1)
                            << "[TensorProtoSerializer::serialize] Expect 1 arguments but get "
                            << args.size();
                        return reinterpret_cast<TensorProtoSerializer*>(self)->Serialize(args[0]);
                      })
    .RegisterFunction("deserialize",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 1)
                            << "[TensorProtoSerializer::deserialize] Expect 1 arguments but get "
                            << args.size();
                        auto view = args[0].As<string_view>();
                        return reinterpret_cast<TensorProtoSerializer*>(self)->Deserialize(view);
                      });

}  // namespace runtime
}  // namespace matxscript
