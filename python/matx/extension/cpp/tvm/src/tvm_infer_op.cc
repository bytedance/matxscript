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
#include "tvm_infer_op.h"

#include <matxscript/pipeline/internal_helper_funcs.h>
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/container/unicode_helper.h>

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(TVMInferOp).SetThreadSafety(false);

void TVMInferOp::Init() {
  // get device
  if (device_ == NONE_DEVICE) {
    if (HasAttr("device")) {
      options_.device = GetAttr<int>("device");
    } else {
      MXLOG(INFO) << "[TVMInferOp] devices not config, use default device: " << options_.device;
      options_.device = -1;
    }
  } else {
    options_.device = device_;
    MXLOG(INFO) << "[TVMInferOp] use devices: " << options_.device;
  }

  options_.share_model = GetAttr<bool>("share_model");
  options_.batch_arg_name = GetAttr<Unicode>("batch_arg_name").encode();
  List models = GetAttr<List>("models");
  for (auto& item : models) {
    auto model_info = item.AsObjectRef<Dict>();
    int batch_size = model_info.get_item(U"batch_size").As<int32_t>();
    String model_name = model_info.get_item(U"model_name").As<Unicode>().encode();
    MXCHECK(!model_name.empty()) << "[TVMInferOp] model_name is empty";
    options_.models.emplace(batch_size, model_name);
  }

  for (auto& item : options_.models) {
    String& model = item.second;
    auto tvm_model = std::dynamic_pointer_cast<TVMModel>(GetOpImpl("TVMModel", model));
    MXCHECK(tvm_model != nullptr) << "can't find model: " << model;
    auto engine = tvm_model->RegisterOrGetEngine(internal::cuda_device_offset(options_.device),
                                                 options_.share_model);
    MXCHECK(engine != nullptr) << "init tvm engine failed!";
    engines_.emplace(item.first, engine);
  }
}

RTValue TVMInferOp::Process(PyArgs inputs) const {
  int batch_size = -1;
  std::vector<std::pair<std::string, const DLTensor*>> tvm_inputs;
  tvm_inputs.reserve(32);
  for (auto& input : inputs) {
    MXCHECK_EQ(input.type_code(), TypeIndex::kRuntimeDict)
        << "[TVMInferOp] input type error, \n"
        << "optional: Dict[bytes, NDArray] or Dict[str, NDArray], \n"
        << "but receive type : " << input.type_code();
    Dict input_dict = input.AsObjectRefNoCheck<Dict>();
    for (auto items : input_dict.items()) {
      MXCHECK_EQ(items.second.type_code(), TypeIndex::kRuntimeNDArray)
          << "[TVMInferOp] input type error, \n"
          << "optional: Dict[bytes, NDArray] or Dict[str, NDArray], \n"
          << "but receive value type : " << items.second.type_code();
      NDArray tx_tsr = items.second.AsObjectRefNoCheck<NDArray>();
      switch (items.first.type_code()) {
        case TypeIndex::kRuntimeString: {
          auto s = items.first.As<String>();
          tvm_inputs.emplace_back(std::string(s.data(), s.size()), tx_tsr.operator->());
        } break;
        case TypeIndex::kRuntimeUnicode: {
          auto u = items.first.As<Unicode>();
          auto s = UnicodeHelper::Encode(u);
          tvm_inputs.emplace_back(std::string(s.data(), s.size()), tx_tsr.operator->());
        } break;
        default: {
          /* not compatible type */
          MXCHECK(false) << "[TVMInferOp] input type error, \n"
                         << "optional: Dict[bytes, NDArray] or Dict[str, NDArray], \n"
                         << "but receive key type : " << items.first.type_code();
        }
      }
      if (tvm_inputs.back().first == options_.batch_arg_name) {
        batch_size = tx_tsr->shape[0];
      }
    }
  }
  auto engine_itr = engines_.find(batch_size);
  MXCHECK(engine_itr != engines_.end()) << "[TVMInferOp] not supported batch size: " << batch_size;

  std::vector<std::pair<std::string, ::tvm::runtime::NDArray>> tvm_outputs;
  engine_itr->second->forward(tvm_inputs, &tvm_outputs);

  Dict result;
  result.reserve(tvm_inputs.size());
  for (auto& item : tvm_outputs) {
    DLContext ctx;
    ctx.device_type = kDLCPU;
    ctx.device_id = 0;
    auto tx_tsr = NDArray::Empty(item.second.Shape(), item.second->dtype, ctx);
    item.second.CopyToBytes(reinterpret_cast<char*>(tx_tsr->data) + tx_tsr->byte_offset,
                            GetDataSize(*tx_tsr.operator->()));
    result[String(std::move(item.first))] = tx_tsr;
  }
  return result;
}

}  // namespace runtime
}  // namespace matxscript
