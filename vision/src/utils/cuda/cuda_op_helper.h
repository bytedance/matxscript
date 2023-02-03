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

#include <cuda_runtime.h>
#include <matxscript/runtime/container/list_ref.h>
#include <matxscript/runtime/container/ndarray.h>
#include <matxscript/runtime/container/opaque_object.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/stream_info.h>

#include "./config.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

typedef enum { ASYNC = 0, SYNC = 1, SYNC_CPU = 2 } VISION_SYNC_MODE;

extern size_t CalculateOutputBufferSize(std::vector<int64_t>& image_shape,
                                        const matxscript::runtime::DataType& data_type);

extern matxscript::runtime::NDArray MakeNDArrayWithWorkSpace(
    const DLDevice& ctx,
    matxscript::runtime::DeviceAPI* cuda_api,
    size_t buffer_size,
    std::vector<int64_t>& image_shape,
    const matxscript::runtime::DataType& data_type,
    std::function<void()> sync_func,
    size_t extra_size,
    void** extra_workspace);

extern MATXScriptStreamHandle GetCudaStreamFromOpaqueObject(
    const matxscript::runtime::OpaqueObject& opaque_object);

extern void check_and_set_device(int device_id);

::matxscript::runtime::NDArray to_cpu(const ::matxscript::runtime::NDArray& image,
                                      MATXScriptStreamHandle stream);

::matxscript::runtime::List to_cpu(const ::matxscript::runtime::List& images,
                                   MATXScriptStreamHandle stream);

::matxscript::runtime::NDArray check_copy(const ::matxscript::runtime::NDArray& image,
                                          DLDevice ctx,
                                          MATXScriptStreamHandle stream);

::matxscript::runtime::List check_copy(const ::matxscript::runtime::List& images,
                                       DLDevice ctx,
                                       MATXScriptStreamHandle stream);

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision