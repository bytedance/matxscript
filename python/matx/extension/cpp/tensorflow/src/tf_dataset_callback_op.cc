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

#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/utf8_util.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tf_utils.h"

namespace {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef tensorflow::gtl::InlinedVector<tensorflow::int64, 4> ShapeContainer;

class MATXScriptTFDatasetCallbackOp : public tensorflow::OpKernel {
 private:
  ::matxscript::runtime::OpKernel* op_impl_ = nullptr;

  void initAttributes(tensorflow::OpKernelConstruction* context) {
    std::string op_addr_s;
    TF_CHECK_OK(context->GetAttr("op_addr", &op_addr_s));
    auto op_addr = std::strtoull(op_addr_s.c_str(), nullptr, 10);
    CHECK(op_addr != 0);
    // TODO: check we need own this ptr
    op_impl_ = static_cast<::matxscript::runtime::OpKernel*>((void*)(op_addr));
  }

 public:
  explicit MATXScriptTFDatasetCallbackOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    // Get attr
    initAttributes(context);
  }

  void Compute(tensorflow::OpKernelContext* context) override;
};

template <typename Iterator>
void SetMultiResult(Iterator first, Iterator last, tensorflow::OpKernelContext* context) {
  for (int i = 0; first != last; ++first, ++i) {
    ::matxscript::runtime::tf_utils::ToTFTensor(
        first->template As<::matxscript::runtime::NDArray>(), context, i);
  }
}

void MATXScriptTFDatasetCallbackOp::Compute(tensorflow::OpKernelContext* context) {
  // the last input is output shape spec
  const int num_inputs = context->num_inputs();
  std::vector<ShapeContainer> shapes(num_inputs);

  // Prepare op inputs
  std::vector<::matxscript::runtime::RTValue> values;
  values.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    // Grab the input tensor
    auto& input_tensor = context->input(i);

    // Create shape container, should keep ref during execution
    shapes[i] = input_tensor.shape().dim_sizes();
    const auto ndims = input_tensor.shape().dims();

    switch (input_tensor.dtype()) {
      case tensorflow::DT_STRING: {
        const auto& flatten = input_tensor.flat<tensorflow::tstring>();
        if (ndims == 0u) {
          auto& value = flatten(0);
          // TODO: zero copy
          // ::matxscript::runtime::string_view value_view(value.data(), value.size());
          // values.emplace_back(::matxscript::runtime::RTView(value_view));
          values.emplace_back(::matxscript::runtime::RTValue(
              ::matxscript::runtime::String(value.data(), value.size())));
        } else if (ndims == 1u) {
          size_t sz = input_tensor.dim_size(0);
          auto list = ::matxscript::runtime::List();
          list.reserve(sz);
          for (size_t i = 0; i < sz; ++i) {
            auto& value_i = flatten(i);
            list.append(::matxscript::runtime::RTValue(
                ::matxscript::runtime::String(value_i.data(), value_i.size())));
          }
          values.emplace_back(std::move(list));
        } else {
          context->SetStatus(tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                                "string tensor only support dim=0/1"));
          return;
        }
      } break;
      case tensorflow::DT_FLOAT: {
        if (ndims == 0u) {
          values.emplace_back(::matxscript::runtime::RTValue(input_tensor.scalar<float>()(0)));
        } else {
          auto tx_tensor = ::matxscript::runtime::tf_utils::FromTFTensor(input_tensor);
          values.emplace_back(std::move(tx_tensor));
        }
      } break;
      case tensorflow::DT_DOUBLE: {
        if (ndims == 0u) {
          values.emplace_back(::matxscript::runtime::RTValue(input_tensor.scalar<double>()(0)));
        } else {
          auto tx_tensor = ::matxscript::runtime::tf_utils::FromTFTensor(input_tensor);
          values.emplace_back(std::move(tx_tensor));
        }
      } break;
      case tensorflow::DT_INT32: {
        if (ndims == 0u) {
          int32_t value = input_tensor.scalar<tensorflow::int32>()(0);
          values.emplace_back(::matxscript::runtime::RTValue(value));
        } else {
          auto tx_tensor = ::matxscript::runtime::tf_utils::FromTFTensor(input_tensor);
          values.emplace_back(std::move(tx_tensor));
        }
      } break;
      case tensorflow::DT_INT64: {
        if (ndims == 0u) {
          int64_t value = input_tensor.scalar<tensorflow::int64>()(0);
          values.emplace_back(::matxscript::runtime::RTValue(value));
        } else {
          auto tx_tensor = ::matxscript::runtime::tf_utils::FromTFTensor(input_tensor);
          values.emplace_back(std::move(tx_tensor));
        }
      } break;
      default: {
        context->SetStatus(tensorflow::Status(
            tensorflow::error::INVALID_ARGUMENT,
            "unsupported tensorflow dtype: " + tensorflow::DataType_Name(input_tensor.dtype())));
        return;
      }
    }
  }

  ::matxscript::runtime::RTValue result;
  try {
    result = op_impl_->Process(::matxscript::runtime::PyArgs(values.data(), values.size()));
  } catch (std::exception& e) {
    context->SetStatus(tensorflow::Status(tensorflow::error::ABORTED, e.what()));
    return;
  } catch (...) {
    context->SetStatus(
        tensorflow::Status(tensorflow::error::ABORTED, "run matxscript op failed!!!"));
    return;
  }

  // output to tensorflow
  switch (result.type_code()) {
    case ::matxscript::runtime::TypeIndex::kRuntimeNDArray: {
      ::matxscript::runtime::tf_utils::ToTFTensor(
          result.AsNoCheck<::matxscript::runtime::NDArray>(), context, 0);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeTuple: {
      auto multi_outputs = result.AsNoCheck<::matxscript::runtime::Tuple>();
      SetMultiResult(multi_outputs.begin(), multi_outputs.end(), context);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeList: {
      auto multi_outputs = result.AsNoCheck<::matxscript::runtime::List>();
      SetMultiResult(multi_outputs.begin(), multi_outputs.end(), context);
    } break;
    default: {
      auto errmsg = "unsupported matxscript result type: " + result.type_name();
      context->SetStatus(tensorflow::Status(tensorflow::error::INTERNAL,
                                            tensorflow::StringPiece(errmsg.data(), errmsg.size())));
      return;
    } break;
  }
  // end
}

REGISTER_KERNEL_BUILDER(Name("MATXScriptTFDatasetCallbackOp").Device(tensorflow::DEVICE_CPU),
                        MATXScriptTFDatasetCallbackOp);

REGISTER_OP("MATXScriptTFDatasetCallbackOp")
    .Input("input_args: ListT")
    .Attr("ListT: list({string, float32, float64, int32, int64})")
    .Output("output: output_dtype")
    .Attr("op_addr: string")
    .Attr("output_dtype: list({float32, int32, int64})");

}  // namespace
