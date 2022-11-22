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

#include <unordered_map>

#include <matxscript/runtime/container/ndarray.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class NDArrayHelper {
 public:
  static bool GetBroadcastShape(const std::vector<int64_t>& shape1,
                                const std::vector<int64_t>& shape2,
                                std::vector<int64_t>& broadcast_shape);
  static bool IsSameShape(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2);
  static bool IsSameShape(const NDArray& nd1, const NDArray& nd2);
  static int64_t GetItemNum(const int64_t* shape, int ndim);
  static DataType DTypePromotion(const DataType& dt1, const DataType& dt2);
  static DataType DTypeFromDouble(const DataType& dt);
  static void* GetData(const NDArray& nd);
  static void IndexesAddOne(const std::vector<int64_t>& shape,
                            size_t dim,
                            std::vector<int64_t>& indexes);
  static DLDevice GetCPUDevice();
  static std::vector<int64_t> ExpandShape(const std::vector<int64_t>& shape, size_t dim);
  static std::vector<int64_t> ExpandStrides(const std::vector<int64_t>& strides, size_t dim);
  static int64_t Offset(const std::vector<int64_t>& indexes,
                        const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& strides);
  static DLDevice GetDevice(const Unicode& device);
  static Unicode GetDeviceStr(const DLDevice& device);
};

class NDArrayOperate {
 public:
  static NDArray Add(const NDArray& lhs, const NDArray& rhs);
  static NDArray Add(const NDArray& lhs, int64_t num);
  static NDArray Add(const NDArray& lhs, double num);

  static NDArray Mul(const NDArray& lhs, const NDArray& rhs);
  static NDArray Mul(const NDArray& lhs, int64_t num);
  static NDArray Mul(const NDArray& lhs, double num);

  static NDArray Sub(const NDArray& lhs, const NDArray& rhs);
  static NDArray Sub(int64_t num, const NDArray& rhs);
  static NDArray Sub(double num, const NDArray& lhs);

  static NDArray Div(const NDArray& lhs, const NDArray& rhs);
  static NDArray Div(double num, const NDArray& rhs);
  static NDArray Div(const NDArray& lhs, double num);

  static NDArray Rand(const std::vector<int64_t>& shape);
  static NDArray Concatenate(const Any& seq, int64_t axis = 0);
  static NDArray Stack(const Any& seq, int64_t axis = 0);
};

}  // namespace runtime
}  // namespace matxscript
