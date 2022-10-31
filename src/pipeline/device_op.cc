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
#include <matxscript/pipeline/device_op.h>
#include "matxscript/pipeline/attributes.h"
#include "matxscript/pipeline/internal_helper_funcs.h"
#include "matxscript/pipeline/tx_session.h"
#include "matxscript/runtime/c_runtime_api.h"
#include "matxscript/runtime/device_api.h"
#include "matxscript/runtime/dlpack.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/threadpool/lock_based_thread_pool.h"

namespace matxscript {
namespace runtime {

void DeviceOp::Init() {
  auto device = GetAttr<Unicode>("device");
  auto it = name2id_.find(device);
  MXCHECK(it != name2id_.end()) << "unsupported device";
  device_id_ = it->second;
}

RTValue DeviceOp::Process(PyArgs inputs) const {
  int session_device_id = device_;
  MATXScriptStreamHandle compute_stream = nullptr, h2d_stream = nullptr, d2h_stream = nullptr;
  void* thread_pool = nullptr;

  if (device_ == NONE_DEVICE) {
    // no session, use global
    if (device_id_ >= 0) {
      MATXScriptContext ctx{kDLGPU, device_id_};
      DeviceAPI* api = DeviceAPI::Get(ctx, true);
      if (api != nullptr) {
        compute_stream = api->GetDefaultComputeStream(ctx);
        h2d_stream = api->GetDefaultIOStreamH2D(ctx);
        d2h_stream = api->GetDefaultIOStreamD2H(ctx);
      }
    }
  } else {
    if (device_ >= 0) {
      session_device_id = internal::cuda_device_offset(device_);
      MATXScriptContext ctx{kDLGPU, session_device_id};
      DeviceAPI* api = DeviceAPI::Get(ctx, true);
      if (api != nullptr) {
        compute_stream = api->GetDefaultComputeStream(ctx);  // use global, TODO: use session local
        h2d_stream = api->GetDefaultIOStreamH2D(ctx);
        d2h_stream = api->GetDefaultIOStreamD2H(ctx);
      }
    }
  }
  thread_pool = belong_to_->GetComputeThreadPool();
  return Dict({{"device_id", device_id_},
               {"session_device_id", session_device_id},
               {"compute_stream", compute_stream},
               {"h2d_stream", h2d_stream},
               {"d2h_stream", d2h_stream},
               {"thread_pool", thread_pool}});
}

std::unordered_map<Unicode, int> DeviceOp::name2id_ = {
    {Unicode(U""), NONE_DEVICE}, {Unicode(U"cpu"), -1},   {Unicode(U"cpu:0"), -1},
    {Unicode(U"gpu:0"), 0},      {Unicode(U"cuda:0"), 0}, {Unicode(U"gpu:1"), 1},
    {Unicode(U"cuda:1"), 1},     {Unicode(U"gpu:2"), 2},  {Unicode(U"cuda:2"), 2},
    {Unicode(U"gpu:3"), 3},      {Unicode(U"cuda:3"), 3}, {Unicode(U"gpu:4"), 4},
    {Unicode(U"cuda:4"), 4},     {Unicode(U"gpu:5"), 5},  {Unicode(U"cuda:5"), 5},
    {Unicode(U"gpu:6"), 6},      {Unicode(U"cuda:6"), 6}, {Unicode(U"gpu:7"), 7},
    {Unicode(U"cuda:7"), 7},     {Unicode(U"cuda:8"), 8}, {Unicode(U"gpu:8"), 8}};

// Device should be rebind for every session
MATX_REGISTER_NATIVE_OP(DeviceOp).SetThreadSafety(false);

}  // namespace runtime
}  // namespace matxscript