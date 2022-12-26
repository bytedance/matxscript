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
#include "torch_infer_op.h"

#include "matxscript/runtime/dlpack.h"
#include "matxscript/runtime/global_type_index.h"
#include "torch_inc.h"

#include <cstdarg>

#include <matxscript/pipeline/internal_helper_funcs.h>
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/container/ndarray.h>
#include <matxscript/runtime/container/ndarray_helper.h>
#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/container/unicode_helper.h>
#include <matxscript/runtime/env_time.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {
// clang-format off
MATX_REGISTER_NATIVE_OP(TorchInferOp).SetThreadSafety(false);
// clang-format on

namespace {

static inline constexpr bool IsCPUMemory(const DLDevice& device) {
  return (device.device_type == kDLCPU) || (device.device_type == kDLCUDAHost);
}

static inline constexpr bool IsCUDAMemory(const DLDevice& device) {
  return (device.device_type == kDLCUDA) || (device.device_type == kDLCUDAManaged);
}

torch::Tensor ToTorchTensor(const NDArray& tx_tsr, at::Device device) {
  if ((IsCUDAMemory(tx_tsr->device) && device.is_cuda()) ||
      (IsCPUMemory(tx_tsr->device) && device.is_cpu())) {
    return at::fromDLPack(tx_tsr.ToDLPack());
  }

  return at::fromDLPack(tx_tsr.ToDLPack()).to(device);
}

NDArray FromTorchTensor(const torch::Tensor& arg_th_tsr, bool output_to_cpu = true) {
  if (!output_to_cpu && arg_th_tsr.get_device() >= 0) {
    if (arg_th_tsr.dim() == 0) {
      return NDArray::FromDLPack(at::toDLPack(arg_th_tsr.unsqueeze(-1)));
    } else {
      return NDArray::FromDLPack(at::toDLPack(arg_th_tsr));
    }
  }
  if (arg_th_tsr.dim() == 0) {
    return NDArray::FromDLPack(at::toDLPack(arg_th_tsr.to(at::kCPU).unsqueeze(-1)));
  } else {
    return NDArray::FromDLPack(at::toDLPack(arg_th_tsr.to(at::kCPU)));
  }
}

}  // namespace

RTValue TorchInferOp::FromIValue(const torch::jit::IValue& i_val) const {
  if (i_val.isInt()) {
    return RTValue(i_val.toInt());
  }
  if (i_val.isDouble()) {
    return RTValue(i_val.toDouble());
  }
  if (i_val.isString()) {
    return RTValue(String(i_val.toString()->string()));
  }
  if (i_val.isTensor()) {
    return RTValue(FromTorchTensor(i_val.toTensor(), output_to_cpu_));
  }
  if (i_val.isIntList()) {
    List ret;
    auto i_list = i_val.toIntList();
    ret.reserve(i_list.size());
    for (auto it = i_list.begin(); it != i_list.end(); ++it) {
      ret.push_back(RTValue(*it));
    }
    return RTValue(ret);
  }
  if (i_val.isDoubleList()) {
    List ret;
    auto i_list = i_val.toDoubleList();
    ret.reserve(i_list.size());
    for (auto it = i_list.begin(); it != i_list.end(); ++it) {
      ret.push_back(RTValue(*it));
    }
    return RTValue(ret);
  }
  if (i_val.isTensorList()) {
    List ret;
    auto i_list = i_val.toTensorList();
    ret.reserve(i_list.size());
    for (auto it = i_list.begin(); it != i_list.end(); ++it) {
      ret.push_back(RTValue(FromTorchTensor(*it, output_to_cpu_)));
    }
    return RTValue(ret);
  }
#ifdef TORCH_1_3
  if (i_val.isGenericList()) {
#else
  if (i_val.isList()) {
#endif
    List ret;
#ifdef TORCH_1_3
    auto i_list = i_val.toGenericList();
#else
    auto i_list = i_val.toList();
#endif
    ret.reserve(i_list.size());
    for (auto it = i_list.begin(); it != i_list.end(); ++it) {
      ret.push_back(FromIValue(*it));
    }
    return RTValue(ret);
  }
  if (i_val.isTuple()) {
    MXCHECK(false) << "Tuple is only supported as first level output";
  }
  if (i_val.isGenericDict()) {
    Dict ret;
    auto dict_ref = i_val.toGenericDict();
    ret.reserve(dict_ref.size());
    for (auto it = dict_ref.begin(); it != dict_ref.end(); ++it) {
      ret.set_item(FromIValue(it->key()), FromIValue(it->value()));
    }
    return RTValue(ret);
  }
  MXCHECK(false) << "can't convert " << *(i_val.type()) << " to RTValue";
  return RTValue();
}

IValueType TorchInferOp::ToIValue(const Any& rt_value) const {
  auto type_code = rt_value.type_code();
  if (type_code == TypeIndex::kRuntimeInteger) {
    return {torch::jit::IValue(rt_value.As<int64_t>()), c10::IntType::get()};
  }
  if (type_code == TypeIndex::kRuntimeFloat) {
    return {torch::jit::IValue(rt_value.As<double>()), c10::FloatType::get()};
  }
  if (type_code == TypeIndex::kRuntimeString) {
    return {torch::jit::IValue(std::string(rt_value.As<string_view>())), c10::StringType::get()};
  }
  if (type_code == TypeIndex::kRuntimeUnicode) {
    auto view = rt_value.As<unicode_view>();
    return {torch::jit::IValue(UTF8Encode(view.begin(), view.size())), c10::StringType::get()};
  }
  if (type_code == TypeIndex::kRuntimeNDArray) {
    auto obj = rt_value.AsObjectViewNoCheck<NDArray>();
    return {torch::jit::IValue(ToTorchTensor(obj.data(), engine_->get_device())),
            c10::TensorType::get()};
  }
  if (type_code == TypeIndex::kRuntimeDict) {
    auto obj = rt_value.AsObjectViewNoCheck<Dict>();
    return ToDict(obj.data());
  }
  if (type_code == TypeIndex::kRuntimeList) {
    auto obj = rt_value.AsObjectViewNoCheck<List>();
    return ToList(obj.data());
  }
  /* 暂不支持Tuple类型作为输入
  if (type_code == TypeIndex::kRuntimeADT)  {
    const ADT& obj = rt_value.AsObjectViewNoCheck<ADT>().data();
    MXCHECK(obj.tag() == 0) << "can't covert ADT[tag=" << obj.tag() << "] to IValue";
    int num = obj.size();
    std::vector<torch::jit::IValue> ivals;
    std::vector<c10::TypePtr> ival_types;
    ivals.reserve(num);
    ival_types.reserve(num);
    for(int i = 0; i < num; ++i) {
      auto ival_type = ToIValue(obj[i]);
      ivals.push_back(ival_type.first);
      ival_types.push_back(ival_type.second);
    }
    return {torch::jit::IValue(c10::ivalue::Tuple::create(ivals)),
  c10::TupleType::create(ival_types)};
  }
  */
  MXCHECK(false) << "can't convert " << TypeIndex2Str(type_code) << " to IValue";
  return {torch::jit::IValue(nullptr), c10::AnyType::get()};
}

c10::TypePtr TorchInferOp::GetIVecTypePtr(const std::vector<IValueType>& i_vec) const {
  if (i_vec.empty()) {
    return c10::AnyType::get();
  }
  c10::TypePtr first_ptr = i_vec[0].second;
  for (auto it = i_vec.begin() + 1; it != i_vec.end(); ++it) {
    c10::TypePtr cur_ptr = it->second;
    if (*cur_ptr == *first_ptr) {
      continue;
    } else {
      return c10::AnyType::get();
    }
  }
  return first_ptr;
}

IValueType TorchInferOp::ToAnyList(const List& rt_list) const {
  c10::TypePtr any_type = c10::AnyType::get();
  c10::impl::GenericList i_list(any_type);
  i_list.reserve(rt_list.size());
  for (auto it = rt_list.begin(); it != rt_list.end(); ++it) {
    i_list.emplace_back(ToIValue(*it).first);
  }
  torch::jit::IValue ret(i_list);
  return {ret, c10::ListType::create(any_type)};
}

IValueType TorchInferOp::ToGenericList(const List& rt_list) const {
  std::vector<IValueType> i_vec;
  i_vec.reserve(rt_list.size());
  for (auto it = rt_list.begin(); it != rt_list.end(); ++it) {
    i_vec.emplace_back(ToIValue(*it));
  }
  c10::TypePtr ele_type = GetIVecTypePtr(i_vec);
  c10::impl::GenericList i_list(ele_type);
  i_list.reserve(i_vec.size());
  for (auto& i_val : i_vec) {
    i_list.emplace_back(std::move(i_val.first));
  }
  torch::jit::IValue ret(i_list);
  return {ret, c10::ListType::create(ele_type)};
}

IValueType TorchInferOp::ToList(const List& rt_list) const {
  if (rt_list.empty()) {
    return ToAnyList(rt_list);
  }
  auto type_code = (rt_list[0]).type_code();
  if (type_code != TypeIndex::kRuntimeInteger && type_code != TypeIndex::kRuntimeFloat &&
      type_code != TypeIndex::kRuntimeNDArray) {
    return ToGenericList(rt_list);
  }
  bool is_same = true;
  for (auto it = rt_list.begin() + 1; it != rt_list.end(); ++it) {
    if (it->type_code() != type_code) {
      is_same = false;
      break;
    }
  }
  if (!is_same) {
    return ToAnyList(rt_list);
  }
  if (type_code == TypeIndex::kRuntimeInteger) {
    c10::List<int64_t> i_list;
    i_list.reserve(rt_list.size());
    for (auto it = rt_list.begin(); it != rt_list.end(); ++it) {
      i_list.emplace_back(it->As<int64_t>());
    }
    torch::jit::IValue ret(i_list);
    return {ret, c10::ListType::create(c10::IntType::get())};
  } else if (type_code == TypeIndex::kRuntimeFloat) {
    c10::List<double> i_list;
    i_list.reserve(rt_list.size());
    for (auto it = rt_list.begin(); it != rt_list.end(); ++it) {
      i_list.emplace_back(it->As<double>());
    }
    torch::jit::IValue ret(i_list);
    return {ret, c10::ListType::create(c10::FloatType::get())};
  } else {
    c10::List<torch::Tensor> i_list;
    i_list.reserve(rt_list.size());
    for (auto it = rt_list.begin(); it != rt_list.end(); ++it) {
      auto obj = it->AsObjectViewNoCheck<NDArray>();
      i_list.emplace_back(ToTorchTensor(obj.data(), engine_->get_device()));
    }
    torch::jit::IValue ret(i_list);
    return {ret, c10::ListType::create(c10::TensorType::get())};
  }
}

IValueType TorchInferOp::ToDict(const Dict& rt_dict) const {
  std::vector<IValueType> keys;
  std::vector<IValueType> vals;
  keys.reserve(rt_dict.size());
  vals.reserve(rt_dict.size());
  for (auto it = rt_dict.begin(); it != rt_dict.end(); ++it) {
    keys.emplace_back(ToIValue(it->first));
    vals.emplace_back(ToIValue(it->second));
  }
  c10::TypePtr key_type = GetIVecTypePtr(keys);
  c10::TypePtr val_type = GetIVecTypePtr(vals);
  c10::impl::GenericDict i_dict(key_type, val_type);
  i_dict.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    i_dict.insert(std::move(keys[i].first), std::move(vals[i].first));
  }
  torch::jit::IValue ret(i_dict);
  return {ret, c10::DictType::create(key_type, val_type)};
}

void TorchInferOp::Init() {
  torch::NoGradGuard no_grad;
  model = GetAttr<Unicode>("model").encode();
  if (HasAttr("output_to_cpu") && GetAttr<int>("output_to_cpu") == 0) {
    output_to_cpu_ = false;
  }
  bool has_device_attr = HasAttr("device");
  int op_device_id = -1;
  if (has_device_attr) {
    auto dev = GetAttr<RTValue>("device", RTValue{-1});
    if (dev.Is<int64_t>()) {
      op_device_id = dev.AsNoCheck<int64_t>();
    } else {
      DLDevice dl_dev;
      if (dev.Is<string_view>()) {
        dl_dev = NDArrayHelper::GetDevice(UTF8Decode(dev.AsNoCheck<string_view>()));
      } else if (dev.Is<unicode_view>()) {
        dl_dev = NDArrayHelper::GetDevice(dev.AsNoCheck<unicode_view>());
      } else {
        THROW_PY_TypeError("invalid device: ", dev);
      }
      if (dl_dev.device_type == DLDeviceType::kDLCPU) {
        op_device_id = -1;
      } else {
        op_device_id = dl_dev.device_id;
      }
    }
  }
  int use_device = -1;
  if (device_ == NONE_DEVICE) {
    if (has_device_attr) {
      use_device = op_device_id;
      MXLOG(INFO) << "[TorchInferOp] use devices: " << use_device;
    }
  } else {
    if (has_device_attr && (op_device_id < 0)) {
      use_device = -1;  // don't use session's device when the mode is on cpu
      MXLOG(INFO) << "[TorchInferOp] use cpu devices: " << use_device;
    } else {
      use_device = device_;
      MXLOG(INFO) << "[TorchInferOp] use session devices: " << use_device;
    }
  }
  th_model_ = std::dynamic_pointer_cast<TorchModel>(belong_to_->FindOp("TorchModel", model));
  MXCHECK(th_model_ != nullptr) << "cnn't find model: " << model;
  sub_ops_ = {th_model_};
  if (use_device >= 0) {
    dl_device_.device_type = kDLCUDA;
    dl_device_.device_id = internal::cuda_device_offset(use_device);
    device_api_ = DeviceAPI::Get(dl_device_);
    engine_ = th_model_->RegisterOrGetEngine(internal::cuda_device_offset(use_device));
  } else {
    dl_device_.device_type = kDLCPU;
    dl_device_.device_id = 0;
    device_api_ = DeviceAPI::Get(dl_device_);
    engine_ = th_model_->RegisterOrGetEngine(use_device);
  }
  MXCHECK(engine_ != nullptr) << "init engine failed!";
}

RTValue TorchInferOp::Process(PyArgs inputs) const {
  torch::NoGradGuard no_grad;
#ifdef MATXSCRIPT_PYTHON_MODE
  if (th_model_ && th_model_->example.is_nullptr()) {
    // for bundle example data
    th_model_->example = Tuple(inputs.begin(), inputs.end());
  }
#endif  // MATXSCRIPT_PYTHON_MODE
#ifdef MATX_ENABLE_TORCH_MODEL_AUTO_SYNCHRONIZATION_WITH_PREPROCESS
  if (dl_device_.device_type == kDLCUDA) {
    cudaStream_t preprocessStream =
        static_cast<cudaStream_t>(device_api_->GetCurrentThreadStream(dl_device_));
    auto torchModelStream = c10::cuda::getCurrentCUDAStream(dl_device_.device_id);
    cudaStream_t torchModelCudaStream = torchModelStream.stream();
    if (preprocessStream != torchModelCudaStream) {
      device_api_->SyncStreamFromTo(dl_device_, preprocessStream, torchModelCudaStream);
    }
  }
#endif
  std::vector<torch::jit::IValue> ival_inputs;
  ival_inputs.reserve(inputs.size());
  torch::jit::IValue ival_output;
  for (auto& rt_val : inputs) {
    ival_inputs.emplace_back(ToIValue(rt_val).first);
  }
  try {
    engine_->forward(ival_inputs, ival_output);
  } catch (const std::exception& e) {
    MXTHROW << "PyTorch forward failed: " << e.what();
  }
  if (ival_output.isTuple()) {
    auto& elements = (ival_output.toTuple())->elements();
    MXCHECK(!elements.empty()) << "model output is empty tuple";
    if (elements.size() == 1) {
      return FromIValue(elements[0]);
    }
    std::vector<RTValue> ret;
    ret.reserve(elements.size());
    for (auto& e : elements) {
      ret.push_back(FromIValue(e));
    }
    return Tuple(ret.begin(), ret.end());
  } else {
    return FromIValue(ival_output);
  }
}

}  // namespace runtime
}  // namespace matxscript
