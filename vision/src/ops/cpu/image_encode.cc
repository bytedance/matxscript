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
#include <opencv2/imgcodecs.hpp>
#include <cstdlib>
#include <exception>
#include <memory>
#include <unordered_map>
#include "matxscript/runtime/c_runtime_api.h"
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/container/tuple_ref.h"
#include "matxscript/runtime/container/unicode.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/device_api.h"
#include "matxscript/runtime/dlpack.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/py_args.h"
#include "matxscript/runtime/runtime_value.h"
#include "matxscript/runtime/threadpool/i_runnable.h"
#include "matxscript/runtime/threadpool/lock_based_thread_pool.h"
#include "ops/base/vision_base_op.h"
#include "utils/object_pool.h"
#include "utils/opencv_util.h"
#include "utils/task_manager.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

namespace {

struct CpuImageEncodeTaskInput {
  CpuImageEncodeTaskInput(NDArray image, const std::vector<int> params, bool is_bgr)
      : image_(std::move(image)), params_(params), is_bgr_(is_bgr) {
  }

  NDArray image_;
  const std::vector<int> params_;
  bool is_bgr_;
};

struct CpuImageEncodeTaskOutput {
  int success = 0;
  std::vector<uchar> binary_data;
};

using CpuImageEncodeTaskInputPtr = std::shared_ptr<CpuImageEncodeTaskInput>;
using CpuImageEncodeTaskOutputPtr = std::shared_ptr<CpuImageEncodeTaskOutput>;

class CpuImageEncodeTask : public internal::LockBasedRunnable {
 public:
  CpuImageEncodeTask(std::vector<CpuImageEncodeTaskInputPtr>::iterator first_input,
                     std::vector<CpuImageEncodeTaskOutputPtr>::iterator first_output,
                     int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<CpuImageEncodeTaskInputPtr>::iterator input_it_;
  std::vector<CpuImageEncodeTaskOutputPtr>::iterator output_it_;
  int len_;
};

void CpuImageEncodeTask::RunImpl() {
  std::vector<CpuImageEncodeTaskInputPtr>::iterator input_it = input_it_;
  std::vector<CpuImageEncodeTaskOutputPtr>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    CpuImageEncodeTaskInputPtr& encode_task_input_ptr = (*input_it);
    CpuImageEncodeTaskOutputPtr& encode_task_output_ptr = (*output_it);
    cv::Mat mat_src;
    if (encode_task_input_ptr->is_bgr_) {
      mat_src = NDArrayToOpencvMat(encode_task_input_ptr->image_);
    } else {
      const cv::Mat& rgb = NDArrayToOpencvMat(encode_task_input_ptr->image_);
      cv::cvtColor(std::move(rgb), mat_src, cv::COLOR_RGB2BGR);
    }
    std::vector<uchar> output_binary;
    bool rt = cv::imencode(".jpg", mat_src, output_binary, encode_task_input_ptr->params_);
    encode_task_output_ptr = std::make_shared<CpuImageEncodeTaskOutput>();
    encode_task_output_ptr->success = rt;
    encode_task_output_ptr->binary_data = std::move(output_binary);
  }
};

}  // namespace

class VisionImencodeOpCPU : public VisionBaseOpCPU {
 public:
  VisionImencodeOpCPU(const Any& session_info,
                      const unicode_view& in_fmt,
                      const int quality,
                      const bool optimized_Huffman);

  ~VisionImencodeOpCPU() = default;
  RTValue process(const List& images, List* flags);

 private:
  int thread_num_ = 0;
  std::vector<int> params_;
  bool is_bgr;  // opencv default is BGR
  TaskManagerPtr task_manager_ptr = nullptr;
};
using CpuImageEncodeTaskInputPtr = std::shared_ptr<CpuImageEncodeTaskInput>;
using CpuImageEncodeTaskOutputPtr = std::shared_ptr<CpuImageEncodeTaskOutput>;

VisionImencodeOpCPU::VisionImencodeOpCPU(const Any& session_info,
                                         const unicode_view& in_fmt,
                                         const int quality,
                                         const bool optimized_Huffman)
    : VisionBaseOpCPU(session_info),
      params_{cv::IMWRITE_JPEG_QUALITY, quality, cv::IMWRITE_JPEG_OPTIMIZE, optimized_Huffman} {
  if (thread_pool_ != nullptr) {
    thread_num_ = thread_pool_->GetThreadsNum();
  }
  if (in_fmt == U"RGB") {
    is_bgr = false;
  } else if (in_fmt == U"BGR") {
    is_bgr = true;
  } else {
    MXTHROW << "Image Encode: output format [" << in_fmt << "] is invalid, please check carefully.";
  }
  task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
}

RTValue VisionImencodeOpCPU::process(const List& images, List* flags) {
  int batch_size = images.size();
  if (batch_size == 0) {
    return List();
  }
  List ret;
  ret.reserve(batch_size);
  if (flags != nullptr) {
    flags->reserve(batch_size);
  }

  std::vector<CpuImageEncodeTaskInputPtr> encode_task_inputs;
  encode_task_inputs.reserve(batch_size);
  for (int i = 0; i < batch_size; i++) {
    auto image = images[i].As<NDArray>();
    encode_task_inputs.emplace_back(
        std::make_shared<CpuImageEncodeTaskInput>(image, params_, is_bgr));
  }

  auto&& encode_task_outputs =
      task_manager_ptr
          ->Execute<CpuImageEncodeTask, CpuImageEncodeTaskInputPtr, CpuImageEncodeTaskOutputPtr>(
              encode_task_inputs, batch_size);

  for (int i = 0; i < batch_size; ++i) {
    int& success = encode_task_outputs[i]->success;
    auto& binary_data = encode_task_outputs[i]->binary_data;
    String bin_str(binary_data.begin(), binary_data.end());
    ret.append(std::move(bin_str));
    if (flags != nullptr) {
      flags->append(success);
    } else {
      MXCHECK(success) << "[image_encode.cc] encoding fail at input index:" << i;
    }
  }
  return ret;
}

class VisionImencodeGeneralOp : public VisionBaseOp {
 public:
  VisionImencodeGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionImencodeOp") {
  }
  ~VisionImencodeGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionImencodeOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 5) << "[VisionImencodeOpCPU] Expect 5 arguments but get "
                                << args.size();
      return std::make_shared<VisionImencodeOpCPU>(
          args[4], args[0].As<unicode_view>(), args[1].As<int>(), args[2].As<bool>());
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 1)
                            << "[VisionImencodeOpCPU][func: process] Expect 1 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionImencodeOpCPU*>(self)->process(
                            args[0].AsObjectView<List>().data(), nullptr);
                      });

using VisionImencodeNoExceptionOpCPU = VisionImencodeOpCPU;
MATX_REGISTER_NATIVE_OBJECT(VisionImencodeNoExceptionOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 5) << "[VisionImencodeOpCPU] Expect 5 arguments but get "
                                << args.size();
      return std::make_shared<VisionImencodeOpCPU>(
          args[4], args[0].As<unicode_view>(), args[1].As<int>(), args[2].As<bool>());
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 1)
                            << "[VisionImencodeOpCPU][func: process] Expect 1 arguments but get "
                            << args.size();
                        List flags;
                        auto ret = reinterpret_cast<VisionImencodeOpCPU*>(self)->process(
                            args[0].AsObjectView<List>().data(), &flags);
                        return Tuple({std::move(ret), std::move(flags)});
                      });

MATX_REGISTER_NATIVE_OBJECT(VisionImencodeGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionImencodeGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionImencodeGeneralOp*>(self)->process(args);
    });

class VisionImencodeNoExceptionOpGeneralOp : public VisionBaseOp {
 public:
  VisionImencodeNoExceptionOpGeneralOp(PyArgs args)
      : VisionBaseOp(args, "VisionImencodeNoExceptionOp") {
  }
  ~VisionImencodeNoExceptionOpGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionImencodeNoExceptionOpGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionImencodeNoExceptionOpGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionImencodeNoExceptionOpGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision