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
#include "matxscript/runtime/data_type.h"
#include "ndarray_helper.h"

namespace byted_matx_vision {
namespace ops {

using namespace ::matxscript::runtime;
namespace {

// general assign
template <typename DstDtype, typename SrcDtype>
void Assign(DstDtype* dst_data,
            const SrcDtype* src_data,
            const int64_t* dst_strides,
            const int64_t* src_strides,
            const int64_t* shape,
            int64_t ndim,
            double alpha,
            double beta) {
  if (ndim == 1) {
    for (int64_t i = 0; i < shape[0]; ++i) {
      dst_data[i * dst_strides[0]] = alpha * src_data[i * src_strides[0]] + beta;
    }
    return;
  }
  for (int64_t i = 0; i < shape[0]; ++i) {
    Assign(dst_data + i * dst_strides[0],
           src_data + i * src_strides[0],
           dst_strides + 1,
           src_strides + 1,
           shape + 1,
           ndim - 1,
           alpha,
           beta);
  }
}

// for compact tensors
template <typename DstDtype, typename SrcDtype>
void Assign(
    DstDtype* dst_data, const SrcDtype* src_data, int64_t element_num, double alpha, double beta) {
  for (int64_t i = 0; i < element_num; ++i) {
    dst_data[i] = alpha * src_data[i] + beta;
  }
}

}  // namespace

NDArray cast_type(const NDArray& input, const Unicode& dtype_str, double alpha, double beta) {
  NDArray::check_dtype_valid(dtype_str);
  DataType dst_dtype(String2DLDataType(UTF8Encode(dtype_str.view())));
  auto ret = NDArray::Empty(input.Shape(), dst_dtype, input->device);
  void* src_data = const_cast<void*>(input.RawData());
  void* dst_data = const_cast<void*>(ret.RawData());
  MATX_NDARRAY_TYPE_SWITCH(dst_dtype, DST_DT, {
    MATX_NDARRAY_TYPE_SWITCH(input.DataType(), SRC_DT, {
      if (input.IsContiguous()) {
        Assign(static_cast<DST_DT*>(dst_data),
               static_cast<SRC_DT*>(src_data),
               input.ElementSize(),
               alpha,
               beta);
      } else {
        Assign(static_cast<DST_DT*>(dst_data),
               static_cast<SRC_DT*>(src_data),
               ret.GetStridesPtr(),
               input.GetStridesPtr(),
               input->shape,
               input->ndim,
               alpha,
               beta);
      }
    });
  });

  return ret;
}

}  // namespace ops
}  // namespace byted_matx_vision