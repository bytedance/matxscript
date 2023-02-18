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

#include <opencv_cuda.h>
#include <matxscript/runtime/data_type.h>
#include <unordered_map>
#include "matxscript/runtime/container/unicode_view.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using CudaDataFormatMap_t =
    std::unordered_map<matxscript::runtime::unicode_view, cuda_op::DataFormat>;

cuda_op::DataType DLDataTypeToOpencvCudaType(matxscript::runtime::DataType dtype);

int UnicodeTODataFormat(const matxscript::runtime::unicode_view& fmt);

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision