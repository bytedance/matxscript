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
#include "tf_infer_op.h"

#include <cstdarg>
#include "matxscript/runtime/global_type_index.h"

#include <matxscript/pipeline/internal_helper_funcs.h>
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/container/unicode_helper.h>
#include <matxscript/runtime/logging.h>

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(TFInferOp).SetThreadSafety(false);

void TFInferOp::Init() {
  if (device_ == NONE_DEVICE) {
    if (HasAttr("device")) {
      options_.device = GetAttr<int>("device");
    }
  } else {
    MXLOG(INFO) << "[TFInferOp] devices not config, use default device: " << options_.device;
    options_.device = device_;
  }

  options_.model = GetAttr<Unicode>("model").encode();
  MXLOG(INFO) << "tensorflow model name " << options_.model;
  auto tf_model = std::dynamic_pointer_cast<TFModel>(belong_to_->FindOp("TFModel", options_.model));
  sub_ops_.push_back(tf_model);
  MXCHECK(tf_model != nullptr) << "can't find model: " << options_.model;
  engine_ = tf_model->RegisterOrGetEngine(internal::cuda_device_offset(options_.device));
  MXCHECK(engine_ != nullptr) << "init tf engine failed!";
}

RTValue TFInferOp::Process(PyArgs inputs) const {
  std::vector<std::pair<std::string, NDArray>> tf_inputs;
  tf_inputs.reserve(32);
  for (auto& input : inputs) {
    MXCHECK_EQ(input.type_code(), TypeIndex::kRuntimeDict)
        << "[TFInferOp] input type error, \n"
        << "optional: Dict[bytes, NDArray] or Dict[str, NDArray], \n"
        << "but receive type : " << input.type_name();
    Dict input_dict = input.AsNoCheck<Dict>();
    for (auto items : input_dict.items()) {
      MXCHECK_EQ(items.second.type_code(), TypeIndex::kRuntimeNDArray)
          << "[TFInferOp] input type error, \n"
          << "optional: Dict[bytes, NDArray] or Dict[str, NDArray], \n"
          << "but receive value type : " << items.second.type_name();
      NDArray tx_tsr = items.second.AsNoCheck<NDArray>();
      switch (items.first.type_code()) {
        case TypeIndex::kRuntimeString: {
          auto s = items.first.AsNoCheck<string_view>();
          tf_inputs.emplace_back(std::string(s.data(), s.size()), tx_tsr);
        } break;
        case TypeIndex::kRuntimeUnicode: {
          auto uv = items.first.AsNoCheck<unicode_view>();
          auto s = UnicodeHelper::Encode(uv);
          tf_inputs.emplace_back(std::string(s.data(), s.size()), tx_tsr);
        } break;
        default: {
          /* not compatible type */
          MXTHROW << "[TFInferOp] input type error, \n"
                  << "optional: Dict[bytes, NDArray] or Dict[str, NDArray], \n"
                  << "but receive key type : " << items.first.type_name();
        }
      }
    }
  }

  return engine_->forward(tf_inputs);
}

}  // namespace runtime
}  // namespace matxscript
