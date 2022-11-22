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
#include "tf_utils.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace matxscript {
namespace runtime {
namespace tf_utils {

static DataType tx_dtype_fp16 = DataType::Float(16, 1);
static DataType tx_dtype_fp32 = DataType::Float(32, 1);
static DataType tx_dtype_int32 = DataType::Int(32, 1);
static DataType tx_dtype_int64 = DataType::Int(64, 1);

static inline tensorflow::DataType tx_dtype_to_tf_dtype(DataType ty) {
  if (ty == tx_dtype_fp16) {
    return tensorflow::DT_HALF;
  } else if (ty == tx_dtype_fp32) {
    return tensorflow::DT_FLOAT;
  } else if (ty == tx_dtype_int32) {
    return tensorflow::DT_INT32;
  } else if (ty == tx_dtype_int64) {
    return tensorflow::DT_INT64;
  }
  return tensorflow::DT_INVALID;
}

static inline DataType tf_dtype_to_tx_dtype(tensorflow::DataType dtype) {
  switch (dtype) {
    case tensorflow::DT_HALF:
      return tx_dtype_fp16;
    case tensorflow::DT_FLOAT:
      return tx_dtype_fp32;
    case tensorflow::DT_INT32:
      return tx_dtype_int32;
    case tensorflow::DT_INT64:
      return tx_dtype_int64;
    default:
      MXTHROW << "Unsupported tensorflow data type " << tensorflow::DataType_Name(dtype) << "("
              << dtype << ")";
  }
  return {};
}

tensorflow::Tensor ToTFTensor(const NDArray& tx_tsr) {
  auto tf_type = tx_dtype_to_tf_dtype(tx_tsr.DataType());
  tensorflow::TensorShape tf_shape;
  for (auto i = 0; i < tx_tsr->ndim; ++i) {
    tf_shape.AddDim(tx_tsr->shape[i]);
  }
  tensorflow::Tensor tf_tsr(tf_type, tf_shape);
  void* tf_ptr_v = const_cast<char*>(tf_tsr.tensor_data().data());
  tx_tsr.CopyToBytes(tf_ptr_v, tf_tsr.AllocatedBytes());
  return tf_tsr;
}

void ToTFTensor(const NDArray& tx_tsr, tensorflow::OpKernelContext* context, int idx) {
  auto num_ele = tx_tsr->ndim <= 0 ? 0 : 1;
  tensorflow::TensorShape tf_shape;
  for (auto i = 0; i < tx_tsr->ndim; ++i) {
    tf_shape.AddDim(tx_tsr->shape[i]);
    num_ele *= tx_tsr->shape[i];
  }

  tensorflow::Tensor* output_tensor;
  OP_REQUIRES_OK(context, context->allocate_output(idx, tf_shape, &output_tensor));
  auto tx_dt = tx_tsr.DataType();
  if (tx_dt == tx_dtype_fp16) {
    auto flat = output_tensor->flat<Eigen::half>();
    static_assert(sizeof(Eigen::half) == sizeof(::matxscript::runtime::Half),
                  "Half not compatible");
    tx_tsr.CopyToBytes(flat.data(), num_ele * sizeof(Eigen::half));
  } else if (tx_dt == tx_dtype_fp32) {
    auto flat = output_tensor->flat<float>();
    tx_tsr.CopyToBytes(flat.data(), num_ele * sizeof(float));
  } else if (tx_dt == tx_dtype_int32) {
    auto flat = output_tensor->flat<tensorflow::int32>();
    tx_tsr.CopyToBytes(flat.data(), num_ele * sizeof(int32_t));
  } else if (tx_dt == tx_dtype_int64) {
    auto flat = output_tensor->flat<tensorflow::int64>();
    tx_tsr.CopyToBytes(flat.data(), num_ele * sizeof(tensorflow::int64));
  } else {
    std::stringstream ss;
    ss << tx_dt;
    context->SetStatus(
        tensorflow::Status(tensorflow::error::INTERNAL, "unsupported NDArray dtype: " + ss.str()));
  }
}

NDArray FromTFTensor(const tensorflow::Tensor& tf_tsr) {
  auto tx_dtype = tf_dtype_to_tx_dtype(tf_tsr.dtype());
  auto& tf_shape = tf_tsr.shape();
  std::vector<int64_t> tx_shape;
  tx_shape.reserve(tf_shape.dims());
  for (auto i = 0; i < tf_shape.dims(); ++i) {
    tx_shape.push_back(tf_shape.dim_size(i));
  }
  DLDevice device;
  device.device_type = kDLCPU;
  device.device_id = 0;
  auto tx_tsr = NDArray::Empty(tx_shape, tx_dtype, device);
  tx_tsr.CopyFromBytes(tf_tsr.tensor_data().data(), tf_tsr.tensor_data().size());
  return tx_tsr;
}

}  // namespace tf_utils
}  // namespace runtime
}  // namespace matxscript
