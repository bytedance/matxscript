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
#include <matxscript/runtime/container/ndarray_helper.h>
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
  auto dl_dev = NDArrayHelper::GetDevice(device);
  if (dl_dev.device_type == kDLCPU) {
    device_id_ = -1;
  } else {
    device_id_ = dl_dev.device_id;
  }
}

RTValue DeviceOp::Process(PyArgs inputs) const {
  int session_device_id = device_;
  MATXScriptStreamHandle current_stream = nullptr;
  void* thread_pool = nullptr;

  if (device_ == NONE_DEVICE) {
    // no session, use global
    if (device_id_ >= 0) {
      MATXScriptDevice ctx{kDLCUDA, device_id_};
      DeviceAPI* api = DeviceAPI::Get(ctx, true);
      if (api != nullptr) {
        current_stream = api->GetCurrentThreadStream(ctx);
      }
    }
  } else {
    if (device_ >= 0) {
      session_device_id = internal::cuda_device_offset(device_);
      MATXScriptDevice ctx{kDLCUDA, session_device_id};
      DeviceAPI* api = DeviceAPI::Get(ctx, true);
      if (api != nullptr) {
        current_stream = api->GetCurrentThreadStream(ctx);
      }
    }
  }
  thread_pool = belong_to_->GetComputeThreadPool();
  // TODO: remove h2d and d2h
  return Dict({{"device_id", device_id_},
               {"session_device_id", session_device_id},
               {"compute_stream", current_stream},
               {"h2d_stream", current_stream},
               {"d2h_stream", current_stream},
               {"thread_pool", thread_pool}});
}

// Device should be rebind for every session
MATX_REGISTER_NATIVE_OP(DeviceOp).SetThreadSafety(false);

}  // namespace runtime
}  // namespace matxscript