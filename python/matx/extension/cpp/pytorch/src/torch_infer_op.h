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

#include "matxscript/runtime/c_runtime_api.h"
#include "torch_model.h"

#include <mutex>

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/device_api.h>

namespace matxscript {
namespace runtime {

using IValueType = std::pair<torch::jit::IValue, c10::TypePtr>;

class TorchInferOp : public OpKernel {
 public:
  void Init() override;
  RTValue Process(PyArgs inputs) const override;

 protected:
  String model;
  std::shared_ptr<TorchModel> th_model_;
  TorchEnginePtr engine_;
  bool output_to_cpu_ = true;
  DeviceAPI* device_api_ = nullptr;
  MATXScriptDevice dl_device_;

 public:
  RTValue FromIValue(const torch::jit::IValue& i_val) const;
  IValueType ToIValue(const Any& rt_val) const;
  IValueType ToAnyList(const List& rt_list) const;
  IValueType ToGenericList(const List& rt_list) const;
  IValueType ToList(const List& rt_list) const;
  IValueType ToDict(const Dict& rt_dict) const;
  c10::TypePtr GetIVecTypePtr(const std::vector<IValueType>& i_vec) const;
};

}  // namespace runtime
}  // namespace matxscript