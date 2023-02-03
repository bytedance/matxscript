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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/ndarray_helper.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/generic/generic_funcs.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArray").set_body([](PyArgs args) -> RTValue {
  RTValue data = args[0].As<RTValue>();
  List shape = args[1].As<List>();
  Unicode dtype_str = args[2].As<Unicode>();
  Unicode device_str = args[3].As<Unicode>();
  return Kernel_NDArray::make(data, shape, dtype_str, device_str);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayToList").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.ToList();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayIsContiguous").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.IsContiguous();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayContiguous").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.Contiguous();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayReshape").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.Reshape(args[1]);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArraySqueeze").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.Squeeze(args[1]);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayUnsqueeze").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.Unsqueeze(args[1].As<int64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayStride").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  const int64_t* strides = data.GetStridesPtr();
  return List(strides, strides + data.GetDim());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayGetItem").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.get_item(args[1].As<RTValue>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArraySetItem").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  data.set_item(args[1].As<RTValue>(), args[2].As<RTValue>());
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayGetSlice").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.get_slice(args[1].As<int64_t>(), args[2].As<int64_t>(), args[3].As<int64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArraySetSlice").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  data.set_slice(args[1].As<int64_t>(), args[2].As<int64_t>(), args[3].As<RTValue>());
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayTranspose").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.transpose(args[1].As<RTValue>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayAsType").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return data.as_type(args[1].As<Unicode>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayAdd").set_body([](PyArgs args) -> RTValue {
  return kernel_nd_module_add(args[0].As<RTView>(), args[1].As<RTView>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArraySub").set_body([](PyArgs args) -> RTValue {
  return kernel_nd_module_sub(args[0].As<RTView>(), args[1].As<RTView>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayDiv").set_body([](PyArgs args) -> RTValue {
  return kernel_nd_module_div(args[0].As<RTView>(), args[1].As<RTView>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayMul").set_body([](PyArgs args) -> RTValue {
  return kernel_nd_module_mul(args[0].As<RTView>(), args[1].As<RTView>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayRand").set_body([](PyArgs args) -> RTValue {
  return kernel_nd_module_rand(args[0].As<RTView>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayConcatenate").set_body([](PyArgs args) -> RTValue {
  return NDArrayOperate::Concatenate(args[0].As<RTValue>(), args[1].As<int64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayStack").set_body([](PyArgs args) -> RTValue {
  return NDArrayOperate::Stack(args[0].As<RTValue>(), args[1].As<int64_t>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayCopyToBytes").set_body([](PyArgs args) -> RTValue {
  void* to = reinterpret_cast<void*>(args[0].As<int64_t>());
  auto view = args[1].AsObjectView<NDArray>();
  auto& nd = view.data();
  MATXScriptStreamHandle stream = reinterpret_cast<MATXScriptStreamHandle>(args[2].As<int64_t>());
  MXCHECK(nd.IsContiguous()) << "NDArrayCopyToBytes: only support contiguous NDArray";
  // DeviceAPI::Get(nd->device)->DefaultComputeStreamSync(nd->device);
  DeviceAPI::Get(nd->device)
      ->CopyDataFromTo(nd->data,
                       nd->byte_offset,
                       to,
                       0,
                       nd.DataSize(),
                       nd->device,
                       nd->device,
                       nd->dtype,
                       stream);
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArrayGetImpl").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  return static_cast<int64_t>(data.GetImpl());
});

// TODO: remove this API later. Python shouldn't be able to set impl. For testing purpose only
MATXSCRIPT_REGISTER_GLOBAL("runtime.NDArraySetImpl").set_body([](PyArgs args) -> RTValue {
  NDArray data = args[0].As<NDArray>();
  int64_t flag = args[1].As<int64_t>();
  data.SetImpl(static_cast<NDArray::Impl>(flag));
  return None;
});

}  // namespace runtime
}  // namespace matxscript
