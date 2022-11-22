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
#include "matx_inc.h"
#include "torch_inc.h"
using namespace ::matxscript::runtime;

namespace {
class TensorAPI {
 public:
  TensorAPI() = default;

  ~TensorAPI() = default;

  NDArray add(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    return NDArray::FromDLPack(at::toDLPack(x_tsr.add(y_tsr)));
  }

  NDArray add_(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    x_tsr.add_(y_tsr);
    return x;
  }

  NDArray sub(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    return NDArray::FromDLPack(at::toDLPack(x_tsr.sub(y_tsr)));
  }

  NDArray sub_(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    x_tsr.sub_(y_tsr);
    return x;
  }

  NDArray div(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    return NDArray::FromDLPack(at::toDLPack(x_tsr.div(y_tsr)));
  }

  NDArray div_(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    x_tsr.div_(y_tsr);
    return x;
  }

  NDArray mul(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    return NDArray::FromDLPack(at::toDLPack(x_tsr.mul(y_tsr)));
  }

  NDArray mul_(const NDArray& x, const NDArray& y) {
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    at::Tensor y_tsr = at::fromDLPack(y.ToDLPack());
    x_tsr.div_(y_tsr);
    return x;
  }

  NDArray stack(PyArgs args) {
    int dim = 0;
    if (args.size() > 1) {
      dim = args[1].As<int>();
    }
    std::vector<at::Tensor> tsr_vec;
    for (auto& nd : args[0].AsObjectView<List>().data()) {
      auto view = nd.AsObjectView<NDArray>();
      tsr_vec.push_back(at::fromDLPack(view.data().ToDLPack()));
    }
    return NDArray::FromDLPack(at::toDLPack(at::stack(tsr_vec, dim)));
  }

  NDArray to_type(const NDArray& x, Unicode type) {
    auto it = str2tsr_type.find(type);
    MXCHECK(it != str2tsr_type.end()) << "to_type failed, unknown type: " << type;
    auto src_code = UTF8Decode(DLDataType2String(x.DataType()));
    if (src_code == type) {
      return x;
    }
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    return NDArray::FromDLPack(at::toDLPack(x_tsr.to(it->second)));
  }

  NDArray to_device(const NDArray& x, Unicode dev) {
    DLDevice nd_device = NDArrayHelper::GetDevice(dev);
    if (nd_device.device_id == (x->device).device_id &&
        nd_device.device_type == (x->device).device_type) {
      return x;
    }
    at::Tensor x_tsr = at::fromDLPack(x.ToDLPack());
    if (nd_device.device_type == DLDeviceType::kDLCPU) {
      return NDArray::FromDLPack(at::toDLPack(x_tsr.to(at::kCPU)));
    } else {
      return NDArray::FromDLPack(
          at::toDLPack(x_tsr.to(at::Device(at::kCUDA, nd_device.device_id))));
    }
  }

  static std::unordered_map<Unicode, c10::ScalarType> str2tsr_type;
};

std::unordered_map<Unicode, c10::ScalarType> TensorAPI::str2tsr_type = {
    {U"float32", torch::kFloat32},
    {U"int32", torch::kInt32},
    {U"int64", torch::kInt64},
    {U"uint8", torch::kUInt8}};

}  // namespace

MATX_REGISTER_NATIVE_OBJECT(TensorAPI)
    .SetConstructor<TensorAPI()>()
    .def("add",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->add(x, y);
         })
    .def("add_",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->add_(x, y);
         })
    .def("sub",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->sub(x, y);
         })
    .def("sub_",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->sub_(x, y);
         })
    .def("mul",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->mul(x, y);
         })
    .def("mul_",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->mul_(x, y);
         })
    .def("div",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->div(x, y);
         })
    .def("div_",
         [](void* self, NDArray x, NDArray y) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->div_(x, y);
         })
    .def("stack",
         [](void* self, PyArgs args) -> RTValue {
           return reinterpret_cast<TensorAPI*>(self)->stack(args);
         })
    .def("to_type",
         [](void* self, NDArray x, Unicode type) -> NDArray {
           return reinterpret_cast<TensorAPI*>(self)->to_type(x, type);
         })
    .def("to_device", [](void* self, NDArray x, Unicode device) -> NDArray {
      return reinterpret_cast<TensorAPI*>(self)->to_device(x, device);
    });