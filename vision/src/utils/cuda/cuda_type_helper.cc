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

#include "cuda_type_helper.h"

#include <matxscript/runtime/logging.h>

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using matxscript::runtime::DataType;

cuda_op::DataType DLDataTypeToOpencvCudaType(DataType dtype) {
  uint8_t type_code = dtype.code();
  uint8_t bits = dtype.bits();

  switch (type_code) {
    case kDLInt: {
      if (bits == 8) {
        return cuda_op::kCV_8S;
      } else if (bits == 16) {
        return cuda_op::kCV_16S;
      } else if (bits == 32) {
        return cuda_op::kCV_32S;
      } else {
        MXLOG(FATAL) << "unknown type " << type_code << " bits: " << bits;
        return cuda_op::kCV_8S;
      }
    } break;
    case kDLUInt: {
      if (bits == 8) {
        return cuda_op::kCV_8U;
      } else if (bits == 16) {
        return cuda_op::kCV_16U;
      } else {
        MXLOG(FATAL) << "unknown type " << type_code << " bits: " << bits;
        return cuda_op::kCV_8U;
      }
    } break;
    case kDLFloat: {
      if (bits == 32) {
        return cuda_op::kCV_32F;
      } else if (bits == 64) {
        return cuda_op::kCV_64F;
      } else if (bits == 16) {
        return cuda_op::kCV_16F;
      }
    } break;
  }
  MXLOG(FATAL) << "unknown type code " << type_code;
  return cuda_op::kCV_32F;
}

int UnicodeTODataFormat(const matxscript::runtime::unicode_view& fmt) {
  static CudaDataFormatMap_t cuda_data_fmt_map = {{U"NCHW", cuda_op::kNCHW},
                                                  {U"NHWC", cuda_op::kNHWC},
                                                  {U"CHW", cuda_op::kCHW},
                                                  {U"HWC", cuda_op::kHWC}};
  int data_format{-1};
  auto it = cuda_data_fmt_map.find(fmt);
  if (it != cuda_data_fmt_map.end()) {
    data_format = it->second;
  } else {
    MXCHECK(false) << "data_format [" << fmt << "] is invalidate, please check carefully.";
  }
  return data_format;
}

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision