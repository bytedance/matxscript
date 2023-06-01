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
#include "utils/cuda/cuda_op_helper.h"
#include "utils/object_pool.h"
#include "utils/opencv_util.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

int dev_malloc(void* ctx, void **ptr, size_t size, cudaStream_t stream){
  MATXScriptDevice * casted_ctx = (MATXScriptDevice *) ctx;
  DeviceAPI* dev_api = matxscript::runtime::DeviceAPI::Get(*casted_ctx);;
  *ptr = dev_api->Alloc(casted_ctx, size);
  return *ptr == nullptr;
}

int dev_free(void* ctx, void *ptr, size_t size, cudaStream_t stream){
  MATXScriptDevice * casted_ctx = (MATXScriptDevice *) ctx;
  DeviceAPI* dev_api = matxscript::runtime::DeviceAPI::Get(*casted_ctx);;
  dev_api->Free(local_ctx, ptr);
  return 0;
}

int pin_malloc(void* ctx, void **ptr, size_t size, cudaStream_t stream){
  MATXScriptDevice * casted_ctx = (MATXScriptDevice *) ctx;
  DeviceAPI* pinned_api = matxscript::runtime::DeviceAPI::Get(*casted_ctx);;
  *ptr = pinned_api->Alloc(casted_ctx, size);
  return *ptr == nullptr;
}

int pin_free(void* ctx, void *ptr, size_t size, cudaStream_t stream){
  MATXScriptDevice * casted_ctx = (MATXScriptDevice *) ctx;
  DeviceAPI* pin_api = matxscript::runtime::DeviceAPI::Get(*casted_ctx);;
  pin_api->Free(local_ctx, ptr);
  return 0;
}


struct DecoderHandlerImpl {
  std::shared_ptr<cuda_op::decode_params_t> params;
  std::shared_ptr<cuda_op::Decoder> decoder;
  MATXScriptStreamHandle stream;  // 每个decoder独立stream
  DeviceAPI* api;
  MATXScriptDevice ctx;
  MATXScriptDevice cpu_ctx{kDLCUDAHost, 0};
  nvjpegPinnedAllocatorV2_t pin_allocator{pin_malloc, pin_free, cpu_ctx};
  nvjpegDevAllocatorV2_t dev_allocator{dev_malloc, dev_free, ctx};
  DecoderHandlerImpl(std::shared_ptr<cuda_op::decode_params_t> arg_params,
                     std::shared_ptr<cuda_op::Decoder> arg_decoder,
                     int device_id)
      : params(std::move(arg_params)), decoder(std::move(arg_decoder)) {
    ctx.device_type = kDLCUDA;
    ctx.device_id = device_id;
    api = DeviceAPI::Get(ctx);
    stream = api->CreateStream(ctx);
  }
  ~DecoderHandlerImpl() {
    api->FreeStream(ctx, stream);
  }

  static std::unique_ptr<DecoderHandlerImpl> build_default(nvjpegOutputFormat_t format,
                                                           int device_id) {
    return build(format, device_id, nullptr);
  }

  static std::unique_ptr<DecoderHandlerImpl> build_crop(nvjpegOutputFormat_t format,
                                                        int device_id,
                                                        const std::pair<float, float>& scale,
                                                        const std::pair<float, float>& ratio) {
    cuda_op::RandomCropGenerator* crop_generator = new cuda_op::RandomCropGenerator(ratio, scale);
    return build(format, device_id, crop_generator);
  }

 private:
  static std::unique_ptr<DecoderHandlerImpl> build(nvjpegOutputFormat_t format,
                                                   int device_id,
                                                   cuda_op::RandomCropGenerator* crop_generator);
};

using HandlerPool = vision::BoundedObjectPool<DecoderHandlerImpl>;

typedef cv::Mat (*DecodeFuncPtr)(const string_view& image_binary);

class VisionImdecodeOpGPU : public VisionBaseOpGPU {
 public:
  VisionImdecodeOpGPU(const Any& session_info, const unicode_view& out_fmt, int pool_size);
  VisionImdecodeOpGPU(const Any& session_info,
                      const unicode_view& out_fmt,
                      int pool_size,
                      const List& scale,
                      const List& ratio);
  RTValue process(const List& images, int sync, List* flags);

  inline std::shared_ptr<HandlerPool> get_handler_pool() {
    return handler_pool_;
  }

  inline DecodeFuncPtr get_cpu_func() {
    return cpu_decode_;
  }

  inline cudaStream_t get_h2d_stream() {
    return getStream();
  }

  inline DLDevice get_ctx() {
    return ctx_;
  }

 private:
  void parse_fmt(const unicode_view& out_fmt);

 private:
  ::matxscript::runtime::internal::IThreadPool* local_thread_pool_ = nullptr;
  int pool_size_;
  nvjpegOutputFormat_t output_format_;
  std::shared_ptr<HandlerPool> handler_pool_;
  DecodeFuncPtr cpu_decode_;
};

namespace {
class DecodeTaskOutput {
 public:
  DecodeTaskOutput() {
    CHECK_CUDA_CALL(cudaEventCreate(&finish_event));
  }
  ~DecodeTaskOutput() {
    CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
  }
  cudaEvent_t finish_event;
  NDArray image;
  void* gpu_workspace = nullptr;
  bool success = true;
};

class ImageDecodeTask : public internal::LockBasedRunnable {
 public:
  ImageDecodeTask(VisionImdecodeOpGPU* op,
                  List::iterator input_first,
                  std::vector<DecodeTaskOutput>::iterator output_first,
                  int len,
                  bool no_throw)
      : op_(op), input_it_(input_first), output_it_(output_first), len_(len), no_throw_(no_throw) {
  }

  static std::vector<internal::IRunnablePtr> build_tasks(
      VisionImdecodeOpGPU* op,
      List::iterator input_first,
      std::vector<DecodeTaskOutput>::iterator output_first,
      int len,
      int thread_num,
      bool no_throw);

  static bool is_jpeg(const char* image_data, size_t size);

 protected:
  void RunImpl() override;
  void decode(List::iterator& input_it,
              std::vector<DecodeTaskOutput>::iterator& output_it,
              std::shared_ptr<DecoderHandlerImpl>& handler);

 private:
  VisionImdecodeOpGPU* op_;
  List::iterator input_it_;
  std::vector<DecodeTaskOutput>::iterator output_it_;
  int len_;
  bool no_throw_;
};

bool ImageDecodeTask::is_jpeg(const char* image_data, size_t size) {
  bool result = true;
  uint8_t* data = (uint8_t*)image_data;
  result &= size >= 4;
  result &=
      (data[0] == 0xff && data[1] == 0xd8 && data[size - 1] == 0xd9 && data[size - 2] == 0xff);
  return result;
}

void ImageDecodeTask::decode(List::iterator& input_it,
                             std::vector<DecodeTaskOutput>::iterator& output_it,
                             std::shared_ptr<DecoderHandlerImpl>& handler) {
  int max_batch_size = 1;
  int channel = 3;
  auto view = input_it->As<string_view>();
  std::vector<char> image_data(view.data(), view.data() + view.size());
  if (!is_jpeg(image_data.data(), image_data.size())) {
    cv::Mat cv_img = op_->get_cpu_func()(string_view(image_data.data(), image_data.size()));
    if (cv_img.data == nullptr) {
      MXTHROW << "[Imdecode] decode image failed";
    }
    output_it->image = OpencvMatToNDArray(cv_img, op_->get_ctx(), op_->get_h2d_stream(), false);
    CHECK_CUDA_CALL(cudaEventRecord(output_it->finish_event, op_->get_h2d_stream()));
    return;
  }
  cuda_op::DecoderBuffers decoder_buffers(max_batch_size);
  decoder_buffers.file_data_[0] = std::move(image_data);
  decoder_buffers.file_len_[0] = view.size();
  decoder_buffers.current_names_[0] = ("");
  cudaStream_t cu_stream = static_cast<cudaStream_t>(handler->stream);
  MXCHECK(handler->decoder->prepareBuffers(decoder_buffers.file_data_,
                                           decoder_buffers.file_len_,
                                           decoder_buffers.current_names_,
                                           decoder_buffers.widths_,
                                           decoder_buffers.heights_,
                                           decoder_buffers.nvjpeg_support_,
                                           decoder_buffers.crop_windows_,
                                           // decoder_buffers.orientations_,
                                           decoder_buffers.iout_,
                                           decoder_buffers.isz_,
                                           *(handler->params)) == EXIT_SUCCESS)
      << "[ImdecodeGPU] call xpref decoder prepareBuffer failed";
  // // need to allocate extra workspace to handle exif image
  // // maybe better to have an interface to tell if image needs orientation
  // if (decoder_buffers.orientations_[0] > 1 && decoder_buffers.orientations_[0] < 9) {
  //   cuda_op::DataShape input_shape, output_shape;
  //   output_shape.N = max_batch_size;
  //   output_shape.C = channel;
  //   output_shape.H = decoder_buffers.heights_[0];
  //   output_shape.W = decoder_buffers.widths_[0];
  //   size_t op_buffer_size =
  //       handler->decoder->calBufferSize(input_shape, output_shape, cuda_op::kCV_8U);
  //   output_it->gpu_workspace = handler->api->Alloc(handler->device, op_buffer_size);
  // }
  output_it->image =
      NDArray::Empty({decoder_buffers.heights_[0], decoder_buffers.widths_[0], channel},
                     DLDataType{kDLUInt, 8, 1},
                     handler->ctx);
  void* p = const_cast<void*>((output_it->image).RawData());
  MXCHECK(handler->decoder->infer(decoder_buffers.file_data_,
                                  decoder_buffers.file_len_,
                                  decoder_buffers.iout_,
                                  decoder_buffers.isz_,
                                  decoder_buffers.widths_,
                                  decoder_buffers.heights_,
                                  decoder_buffers.nvjpeg_support_,
                                  decoder_buffers.crop_windows_,
                                  // decoder_buffers.orientations_,
                                  channel,
                                  output_it->gpu_workspace,
                                  &p,
                                  *(handler->params),
                                  cu_stream) == EXIT_SUCCESS)
      << "[ImdecodeGPU] call xpref decoder infer failed";
  CHECK_CUDA_CALL(cudaEventRecord(output_it->finish_event, cu_stream));
}

void ImageDecodeTask::RunImpl() {
  List::iterator input_it = input_it_;
  std::vector<DecodeTaskOutput>::iterator output_it = output_it_;
  auto handler = (op_->get_handler_pool())->borrow();
  if (!no_throw_) {
    for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
      decode(input_it, output_it, handler);
    }
  } else {
    for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
      try {
        decode(input_it, output_it, handler);
      } catch (...) {
        output_it->success = false;
        output_it->image = NDArray::Empty({8, 8, 3}, DLDataType{kDLUInt, 8, 1}, handler->ctx);
      }
    }
  }
}

std::vector<internal::IRunnablePtr> ImageDecodeTask::build_tasks(
    VisionImdecodeOpGPU* op,
    List::iterator input_first,
    std::vector<DecodeTaskOutput>::iterator output_first,
    int len,
    int thread_num,
    bool no_throw) {
  std::vector<internal::IRunnablePtr> ret;
  if (len <= thread_num) {
    ret.reserve(len);
    for (int i = 0; i < len; ++i) {
      ret.emplace_back(
          std::make_shared<ImageDecodeTask>(op, input_first + i, output_first + i, 1, no_throw));
    }
    return ret;
  }

  ret.reserve(thread_num);
  int step = len / thread_num;
  int remainder = len % thread_num;
  for (int i = 0; i < remainder; ++i) {
    ret.emplace_back(
        std::make_shared<ImageDecodeTask>(op, input_first, output_first, step + 1, no_throw));
    input_first += step + 1;
    output_first += step + 1;
  }
  for (int i = remainder; i < thread_num; ++i) {
    ret.emplace_back(
        std::make_shared<ImageDecodeTask>(op, input_first, output_first, step, no_throw));
    input_first += step;
    output_first += step;
  }
  return ret;
}

cv::Mat CpuDecodeBGR(const string_view& image_binary) {
  cv::Mat image;
  cv::Mat opencv_input(1, image_binary.size(), CV_8UC1, (void*)image_binary.data());
  image = cv::imdecode(opencv_input, cv::IMREAD_COLOR);
  return image;
}

cv::Mat CpuDecodeRGB(const string_view& image_binary) {
  cv::Mat decode_image = CpuDecodeBGR(image_binary);
  if (decode_image.data != nullptr) {
    cv::Mat image;
    cv::cvtColor(decode_image, image, cv::COLOR_BGR2RGB);
    return image;
  } else {
    return decode_image;
  }
}

}  // namespace

std::unique_ptr<DecoderHandlerImpl> DecoderHandlerImpl::build(
    nvjpegOutputFormat_t fmt, int device_id, cuda_op::RandomCropGenerator* crop_generator) {
  cuda_op::DataShape max_input_shape, max_output_shape;
  int max_batch_size = 1;
  max_output_shape.N = max_batch_size;
  max_output_shape.C = 3;
  max_output_shape.H = 1024;
  max_output_shape.W = 1024;

  auto decoder =
      std::make_shared<cuda_op::Decoder>(max_input_shape, max_output_shape, crop_generator);
  auto params_deleter = [decoder, crop_generator](cuda_op::decode_params_t* params_ptr) {
    decoder->destoryParams(*params_ptr);
    delete params_ptr;
    if (crop_generator != nullptr)
      delete crop_generator;
  };
  auto params =
      std::shared_ptr<cuda_op::decode_params_t>(new cuda_op::decode_params_t, params_deleter);
  decoder->prepareDecoderParams("", max_batch_size, fmt, *params, &dev_allocator, &pin_allocator);
  auto ptr = std::make_unique<DecoderHandlerImpl>(std::move(params), std::move(decoder), device_id);
  return ptr;
}

VisionImdecodeOpGPU::VisionImdecodeOpGPU(const Any& session_info,
                                         const unicode_view& out_fmt,
                                         int pool_size)
    : VisionBaseOpGPU(session_info) {
  MXCHECK(pool_size > 1)
      << "[VisionImdecodeOpGPU] pool size must be greater then one and power of 2";
  MXCHECK_EQ((pool_size & (pool_size - 1)), 0)
      << "[VisionImdecodeOpGPU] pool size must be greater then one and power of 2";
  local_thread_pool_ =
      new ::matxscript::runtime::internal::LockBasedThreadPool(pool_size, "ImdecodeThreadPool");
  pool_size_ = pool_size;
  cv::setNumThreads(0);
  parse_fmt(out_fmt);
  std::vector<std::unique_ptr<DecoderHandlerImpl>> handlers;
  handlers.reserve(pool_size_);
  for (int i = 0; i < pool_size_; ++i) {
    handlers.push_back(DecoderHandlerImpl::build_default(output_format_, device_id_));
  }
  handler_pool_ = std::make_shared<HandlerPool>(std::move(handlers));
}

VisionImdecodeOpGPU::VisionImdecodeOpGPU(const Any& session_info,
                                         const unicode_view& out_fmt,
                                         int pool_size,
                                         const List& scale,
                                         const List& ratio)
    : VisionBaseOpGPU(session_info) {
  MXCHECK(pool_size > 1)
      << "[VisionImdecodeOpGPU] pool size must be greater then one and power of 2";
  MXCHECK_EQ((pool_size & (pool_size - 1)), 0)
      << "[VisionImdecodeOpGPU] pool size must be greater then one and power of 2";
  local_thread_pool_ =
      new ::matxscript::runtime::internal::LockBasedThreadPool(pool_size, "ImdecodeThreadPool");
  pool_size_ = pool_size;
  cv::setNumThreads(0);
  parse_fmt(out_fmt);
  if (scale.size() != 2 || scale[0].type_code() != TypeIndex::kRuntimeFloat ||
      scale[1].type_code() != TypeIndex::kRuntimeFloat) {
    MXTHROW << "image crop decode: invalid scale input";
  }
  if (ratio.size() != 2 || ratio[0].type_code() != TypeIndex::kRuntimeFloat ||
      ratio[1].type_code() != TypeIndex::kRuntimeFloat) {
    MXTHROW << "image crop decode: invalid ratio input";
  }
  std::pair<float, float> ratio_pair({ratio[0].As<float>(), ratio[1].As<float>()});
  std::pair<float, float> scale_pair({scale[0].As<float>(), scale[1].As<float>()});
  std::vector<std::unique_ptr<DecoderHandlerImpl>> handlers;
  handlers.reserve(pool_size_);
  for (int i = 0; i < pool_size_; ++i) {
    handlers.push_back(
        DecoderHandlerImpl::build_crop(output_format_, device_id_, scale_pair, ratio_pair));
  }
  handler_pool_ = std::make_shared<HandlerPool>(std::move(handlers));
}

RTValue VisionImdecodeOpGPU::process(const List& images, int sync, List* flags) {
  // prepare input & output
  if (images.size() == 0) {
    return List();
  }
  std::vector<DecodeTaskOutput> outputs(images.size());

  // build task & run
  // ImageDecodeTask task(handler_pool_.get(), image_list.begin(), outputs.begin(), outputs.size());
  // task.Run();

  bool no_throw = (flags != nullptr);
  auto tasks = ImageDecodeTask::build_tasks(
      this, images.begin(), outputs.begin(), outputs.size(), pool_size_, no_throw);
  for (size_t i = 0; i < tasks.size(); ++i) {
    local_thread_pool_->Enqueue(tasks[i], 0);
  }

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

  // syc event
  List ret;
  ret.reserve(images.size());

  if (flags != nullptr) {
    flags->reserve(images.size());
    for (auto& output : outputs) {
      flags->push_back(output.success);
    }
  }

  for (auto& output : outputs) {
    if (output.success) {
      cudaEventSynchronize(output.finish_event);
    }
    ret.push_back(std::move(output.image));
    if (output.gpu_workspace != nullptr) {
      cuda_api_->Free(ctx_, output.gpu_workspace);
    }
  }

  if (sync == VISION_SYNC_MODE::SYNC_CPU) {
    return to_cpu(ret, getStream());
  }
  return ret;
}

void VisionImdecodeOpGPU::parse_fmt(const unicode_view& fmt) {
  if (fmt == U"RGB") {
    output_format_ = NVJPEG_OUTPUT_RGBI;
    cpu_decode_ = CpuDecodeRGB;
    return;
  }
  if (fmt == U"BGR") {
    output_format_ = NVJPEG_OUTPUT_BGRI;
    cpu_decode_ = CpuDecodeBGR;
    return;
  }
  MXTHROW << "Image Decode: output format [" << fmt << "] is invalid, please check carefully.";
}

MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 3) << "[VsionImdecodeOpGPU] Expect 3 arguments but get "
                                 << args.size();
      return std::make_shared<VisionImdecodeOpGPU>(
          args[2], args[0].As<unicode_view>(), args[1].As<int>());
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 2) << "[VsionImdecodeOpGPU] Expect 2 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionImdecodeOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(), args[1].As<int>(), nullptr);
    });

using VisionImdecodeRandomCropOpGPU = VisionImdecodeOpGPU;
MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeRandomCropOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 5) << "[VsionImdecodeCropOpGPU] Expect 5 arguments but get "
                                 << args.size();
      return std::make_shared<VisionImdecodeOpGPU>(args[4],
                                                   args[0].As<unicode_view>(),
                                                   args[3].As<int>(),
                                                   args[1].AsObjectView<List>().data(),
                                                   args[2].AsObjectView<List>().data());
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 2) << "[VsionImdecodeCropOpGPU] Expect 2 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionImdecodeOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(), args[1].As<int>(), nullptr);
    });

using VisionImdecodeNoExceptionOpGPU = VisionImdecodeOpGPU;
MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeNoExceptionOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 3) << "[VsionImdecodeOpGPU] Expect 3 arguments but get "
                                 << args.size();
      return std::make_shared<VisionImdecodeOpGPU>(
          args[2], args[0].As<unicode_view>(), args[1].As<int>());
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 2) << "[VsionImdecodeOpGPU] Expect 2 arguments but get "
                                 << args.size();
      List flags;
      auto ret = reinterpret_cast<VisionImdecodeOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(), args[1].As<int>(), &flags);
      return Tuple({ret, flags});
    });

using VisionImdecodeNoExceptionRandomCropOpGPU = VisionImdecodeOpGPU;
MATX_REGISTER_NATIVE_OBJECT(VisionImdecodeNoExceptionRandomCropOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 5)
          << "[VisionImdecodeNoExceptionRandomCropOpGPU] Expect 5 arguments but get "
          << args.size();
      return std::make_shared<VisionImdecodeOpGPU>(args[4],
                                                   args[0].As<unicode_view>(),
                                                   args[3].As<int>(),
                                                   args[1].AsObjectView<List>().data(),
                                                   args[2].AsObjectView<List>().data());
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 2)
          << "[VisionImdecodeNoExceptionRandomCropOpGPU] Expect 2 arguments but get "
          << args.size();
      List flags;
      auto ret = reinterpret_cast<VisionImdecodeOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(), args[1].As<int>(), &flags);
      return Tuple({ret, flags});
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision
