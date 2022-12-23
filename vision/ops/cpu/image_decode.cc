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
#include <exception>
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/py_args.h"
#include "matxscript/runtime/runtime_value.h"
#include "matxscript/runtime/threadpool/lock_based_thread_pool.h"
#include "ops/base/vision_base_op.h"
#include "ops/cpu/vision_base_op_cpu.h"
#include "utils/opencv_util.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

namespace {
using DecodeFunction = std::function<RTValue(const string_view& image_binary)>;

RTValue DecodeBGR(const string_view& image_binary) {
  cv::Mat image;
  cv::Mat opencv_input(1, image_binary.size(), CV_8UC1, (void*)image_binary.data());
  image = cv::imdecode(opencv_input, cv::IMREAD_COLOR);
  if (image.data == nullptr) {
    MXTHROW << "[Imdecode] decode image failed";
  }
  // convert to NDArray
  return OpencvMatToNDArray(image);
}

DecodeFunction GetDecodeCVTFunction(int code) {
  return [code](const string_view& image_binary) -> NDArray {
    cv::Mat image;
    cv::Mat opencv_input(1, image_binary.size(), CV_8UC1, (void*)image_binary.data());
    cv::Mat decode_image = cv::imdecode(opencv_input, cv::IMREAD_COLOR);
    if (decode_image.data == nullptr) {
      MXTHROW << "[Imdecode] decode image failed";
    }
    cv::cvtColor(decode_image, image, code);
    return OpencvMatToNDArray(image);
  };
}

class ImageDecodeTask : public internal::LockBasedRunnable {
 public:
  ImageDecodeTask(DecodeFunction* decode_func,
                  List::iterator input_first,
                  List::iterator output_first,
                  int len)
      : decode_func_(decode_func), input_it_(input_first), output_it_(output_first), len_(len) {
  }

  static std::vector<internal::IRunnablePtr> build_tasks(DecodeFunction* decode_func_,
                                                         List::iterator input_first,
                                                         List::iterator output_first,
                                                         int len,
                                                         int thread_num);

 protected:
  void RunImpl() override;

 private:
  DecodeFunction* decode_func_;
  List::iterator input_it_;
  List::iterator output_it_;
  int len_;
};

void ImageDecodeTask::RunImpl() {
  for (int i = 0; i < len_; ++i) {
    *(output_it_ + i) = (*decode_func_)((input_it_ + i)->As<string_view>());
  }
}

std::vector<internal::IRunnablePtr> ImageDecodeTask::build_tasks(DecodeFunction* decode_func,
                                                                 List::iterator input_first,
                                                                 List::iterator output_first,
                                                                 int len,
                                                                 int thread_num) {
  std::vector<internal::IRunnablePtr> ret;
  if (len <= thread_num) {
    ret.reserve(len);
    for (int i = 0; i < len; ++i) {
      ret.emplace_back(
          std::make_shared<ImageDecodeTask>(decode_func, input_first + i, output_first + i, 1));
    }
    return ret;
  }

  ret.reserve(thread_num);
  int step = len / thread_num;
  int remainder = len % thread_num;
  for (int i = 0; i < remainder; ++i) {
    ret.emplace_back(
        std::make_shared<ImageDecodeTask>(decode_func, input_first, output_first, step + 1));
    input_first += step + 1;
    output_first += step + 1;
  }
  for (int i = remainder; i < thread_num; ++i) {
    ret.emplace_back(
        std::make_shared<ImageDecodeTask>(decode_func, input_first, output_first, step));
    input_first += step;
    output_first += step;
  }
  return ret;
}

}  // namespace

class VisionImdecodeOpCPU : public VisionBaseOpCPU {
 public:
  VisionImdecodeOpCPU(const Any& session_info, const unicode_view& fmt)
      : VisionBaseOpCPU(session_info) {
    if (fmt == U"BGR") {
      decode_func_ = DecodeBGR;
    } else if (fmt == U"RGB") {
      decode_func_ = GetDecodeCVTFunction(cv::COLOR_BGR2RGB);
    } else {
      MXTHROW << "[ImdecodeOp]: unspported format:" << fmt;
    }
    if (thread_pool_ != nullptr) {
      thread_num_ = thread_pool_->GetThreadsNum();
    }
  }

  RTValue process(const List& images) {
    cv::setNumThreads(0);
    if (images.size() == 0) {
      return List();
    }
    List ret(images.size(), None);
    auto tasks = ImageDecodeTask::build_tasks(
        &decode_func_, images.begin(), ret.begin(), images.size(), thread_num_ + 1);

    for (size_t i = 1; i < tasks.size(); ++i) {
      thread_pool_->Enqueue(tasks[i], 0);
    }
    tasks[0]->Run();
    std::exception_ptr eptr;
    for (size_t i = 0; i < tasks.size(); ++i) {
      try {
        tasks[i]->Wait();
      } catch (...) {
        if (!eptr) {
          // store first exception
          eptr = std::current_exception();
        }
      }
    }
    if (eptr) {
      std::rethrow_exception(eptr);
    }
    return ret;
  }

 private:
  int thread_num_ = 0;
  DecodeFunction decode_func_;
};

MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      // only use two arguments, args[1] is decoder pool size, with is only used in gpu
      MXCHECK(args.size() == 3) << "[VsionImdecodeOpCPU] Expect 3 arguments but get "
                                << args.size();
      return std::make_shared<VisionImdecodeOpCPU>(args[2], args[0].As<unicode_view>());
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK(args.size() == 2) << "[VsionImdecodeOpCPU][func: process] Expect 2 arguments but get "
                                << args.size();
      return reinterpret_cast<VisionImdecodeOpCPU*>(self)->process(
          args[0].AsObjectView<List>().data());
    });

class VisionImdecodeGeneralOp : public VisionBaseOp {
 public:
  VisionImdecodeGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionImdecodeOp") {
  }
  ~VisionImdecodeGeneralOp() = default;
};

class VisionImdecodeRandomCropGeneralOp : public VisionBaseOp {
 public:
  VisionImdecodeRandomCropGeneralOp(PyArgs args)
      : VisionBaseOp(args, "VisionImdecodeRandomCropOp") {
  }
  ~VisionImdecodeRandomCropGeneralOp() = default;
};

class VisionImdecodeNoExceptionGeneralOp : public VisionBaseOp {
 public:
  VisionImdecodeNoExceptionGeneralOp(PyArgs args)
      : VisionBaseOp(args, "VisionImdecodeNoExceptionOp") {
  }
  ~VisionImdecodeNoExceptionGeneralOp() = default;
};

class VisionImdecodeNoExceptionRandomCropGeneralOp : public VisionBaseOp {
 public:
  VisionImdecodeNoExceptionRandomCropGeneralOp(PyArgs args)
      : VisionBaseOp(args, "VisionImdecodeNoExceptionRandomCropOp") {
  }
  ~VisionImdecodeNoExceptionRandomCropGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionImdecodeGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionImdecodeGeneralOp*>(self)->process(args);
    });

MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeRandomCropGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionImdecodeRandomCropGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionImdecodeRandomCropGeneralOp*>(self)->process(args);
    });

MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeNoExceptionGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionImdecodeNoExceptionGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionImdecodeNoExceptionGeneralOp*>(self)->process(args);
    });

MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeNoExceptionRandomCropGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionImdecodeNoExceptionRandomCropGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionImdecodeNoExceptionRandomCropGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision