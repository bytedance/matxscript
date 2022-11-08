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

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/dlpack.h>
#include <matxscript/runtime/generic/generic_funcs.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

// set device api
MATXSCRIPT_REGISTER_GLOBAL("runtime.SetDevice").set_body([](PyArgs args) -> RTValue {
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(args[0].As<int64_t>());
  ctx.device_id = args[1].As<int64_t>();
  DeviceAPI::Get(ctx)->SetDevice(ctx);
  return None;
});

// set device api
MATXSCRIPT_REGISTER_GLOBAL("runtime.GetDeviceAttr").set_body([](PyArgs args) -> RTValue {
  MATXScriptContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(args[0].As<int64_t>());
  ctx.device_id = args[1].As<int64_t>();

  RTValue ret;
  DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].As<int64_t>());
  if (kind == kExist) {
    DeviceAPI* api = DeviceAPI::Get(ctx, true);
    if (api != nullptr) {
      api->GetAttr(ctx, kind, &ret);
    } else {
      ret = 0;
    }
  } else {
    DeviceAPI::Get(ctx)->GetAttr(ctx, kind, &ret);
  }
  return ret;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.MATXScriptSetCurrentThreadStream")
    .set_body_typed(MATXScriptSetCurrentThreadStream);

// create stream
MATXSCRIPT_REGISTER_GLOBAL("runtime.DefaultStream").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "cuda_module_default_stream expect 1 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeInteger)
      << "Create Stream first arg must be integer. ";
  int device_id = args[0].As<int64_t>();
  MXCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  return kernel_cuda_module_create_stream(device_id);
});

// create stream
MATXSCRIPT_REGISTER_GLOBAL("runtime.CreateStream").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "cuda_module_create_stream expect 1 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeInteger)
      << "Create Stream first arg must be integer. ";
  int device_id = args[0].As<int64_t>();
  MXCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  return kernel_cuda_module_create_stream(device_id);
});

// StreamSync
MATXSCRIPT_REGISTER_GLOBAL("runtime.StreamSync").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2) << "StreamSync expect 2 args, bug get " << args.size();
  RTView opaq = args[0].As<RTView>();
  int device_id = args[1].As<int64_t>();
  MXCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  kernel_cuda_module_stream_sync(opaq, device_id);
  return None;
});

// StreamSync
MATXSCRIPT_REGISTER_GLOBAL("runtime.CurrentThreadStreamSync").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "CurrentThreadStreamSync expect 1 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeInteger)
      << "CurrentThreadStreamSync first arg must be integer. ";
  int device_id = args[0].As<int64_t>();
  if (device_id >= 0) {
    MATXScriptContext ctx{kDLGPU, device_id};
    DeviceAPI::Get(ctx)->CurrentThreadStreamSync(ctx);
  }
  return None;
});

}  // namespace runtime
}  // namespace matxscript
