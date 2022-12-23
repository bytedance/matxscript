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

#include <opencv2/opencv.hpp>

#include <matxscript/runtime/container/ndarray.h>
#include <matxscript/runtime/dlpack.h>
#include "matxscript/runtime/c_runtime_api.h"

namespace byted_matx_vision {
namespace ops {

cv::Mat NDArrayToOpencvMat(const matxscript::runtime::NDArray& ndarray);
matxscript::runtime::NDArray OpencvMatToNDArray(const cv::Mat& mat,
                                                DLDevice ctx = {DLDeviceType::kDLCPU, 0},
                                                MATXScriptStreamHandle stream = nullptr,
                                                bool sync = true);
int UnicodeToOpencvInterp(matxscript::runtime::unicode_view opencv_interp);
int UnicodeToOpencvColorCode(matxscript::runtime::unicode_view color_code);

}  // namespace ops
}  // namespace byted_matx_vision