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
#include "utils/cuda/cuda_type_helper.h"
#include "utils/object_pool.h"
#include "utils/opencv_util.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

class EncoderHandlerImpl {
 public:
  static std::unique_ptr<EncoderHandlerImpl> build(cuda_op::Encoder::handle_sharedPtr nvjpeg_handle,
                                           cuda_op::Encoder::param_sharedPtr param,
                                            cuda_op::DataShape shape,
                                           int device_id) {
    auto encoder_ptr = std::make_shared<cuda_op::Encoder>(shape, shape, std::move(nvjpeg_handle), std::move(param));
    auto ptr = std::make_unique<EncoderHandlerImpl>(std::move(encoder_ptr), device_id);
    return std::move(ptr);
  }

  EncoderHandlerImpl(std::shared_ptr<cuda_op::Encoder> arg_encoder,
              int device_id)
        : encoder(std::move(arg_encoder)) {
    ctx.device_type = kDLCUDA;
    ctx.device_id = device_id;
    api = DeviceAPI::Get(ctx);
    stream = api->CreateStream(ctx);
    encoder->createState(static_cast<cudaStream_t>(stream));
  }

  EncoderHandlerImpl(){
    api->FreeStream(ctx, stream);
  }

  ~EncoderHandlerImpl() {
    api->FreeStream(ctx, stream);
  }

  std::shared_ptr<cuda_op::Encoder> encoder;
  MATXScriptStreamHandle stream;  // 每个encoder独立stream
  DeviceAPI* api;
  MATXScriptDevice ctx;
};

using HandlerPool = vision::BoundedObjectPool<EncoderHandlerImpl>;

class VisionImencodeOpGPU : public VisionBaseOpGPU {
 public:
  VisionImencodeOpGPU(const Any& session_info, const unicode_view& in_fmt, const int quality, const bool optimized_Huffman, int pool_size);

  RTValue process(const List& arg_images, List* flags);

  inline std::shared_ptr<HandlerPool> get_handler_pool() {
    return handler_pool_;
  }

  inline cudaStream_t get_h2d_stream() {
    return getStream();
  }

  inline DLDevice get_ctx() {
    return ctx_;
  }

 private:
  static nvjpegInputFormat_t parse_fmt(const unicode_view& in_fmt);

 private:
  ::matxscript::runtime::internal::IThreadPool* local_thread_pool_ = nullptr;
  int pool_size_;
  std::shared_ptr<HandlerPool> handler_pool_;
 
 public:
  nvjpegInputFormat_t input_format_;
  cuda_op::Encoder::handle_sharedPtr nvjpeg_handle;
  cuda_op::Encoder::param_sharedPtr nvjpeg_param;
};

namespace {

class EncodeTaskOutput {
 public:
  EncodeTaskOutput() {
    CHECK_CUDA_CALL(cudaEventCreate(&finish_event));
  }
  ~EncodeTaskOutput() {
    CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
  }
  cudaEvent_t finish_event;
  matxscript::runtime::String image_binary;
  bool success = true;
};

class ImageEncodeTask : public internal::LockBasedRunnable {
public:
     ImageEncodeTask(VisionImencodeOpGPU* op,
                  List::iterator input_first,
                  std::vector<EncodeTaskOutput>::iterator output_first,
                  int len,
                  bool no_throw)
      : op_(op), input_it_(input_first), output_it_(output_first), len_(len), no_throw_(no_throw) {
  }

  static std::vector<internal::IRunnablePtr> build_tasks(
      VisionImencodeOpGPU* op,
      List::iterator input_first,
      std::vector<EncodeTaskOutput>::iterator output_first,
      int len,
      int thread_num,
      bool no_throw);

protected:
  void RunImpl() override;
  void encode(List::iterator& input_it,
              std::vector<EncodeTaskOutput>::iterator& output_it,
              std::shared_ptr<EncoderHandlerImpl>& handler);

private:
  VisionImencodeOpGPU* op_;
  List::iterator input_it_;
  std::vector<EncodeTaskOutput>::iterator output_it_;
  int len_;
  bool no_throw_;
};

void ImageEncodeTask::encode(List::iterator& input_it,
                             std::vector<EncodeTaskOutput>::iterator& output_it,
                             std::shared_ptr<EncoderHandlerImpl>& handler) {

    auto view_elem = input_it->AsObjectView<NDArray>();
    const NDArray& input_nd = view_elem.data();

    std::vector<int64_t> &&src_shape = input_nd.Shape();

    MXCHECK(src_shape[2]==3) << "The inputs must have 3 channels";

    cuda_op::DataShape shape;
    shape.N = 1;
    shape.C = 3;
    shape.H = src_shape[0];
    shape.W = src_shape[1];


    auto data_type = DLDataTypeToOpencvCudaType(input_nd.DataType());
    MXCHECK(data_type==cuda_op::kCV_8U || data_type == cuda_op::kCV_8S)<< "The data type of the inputs must be uint8 or int8";


    cudaStream_t cu_stream  = static_cast<cudaStream_t>(handler->stream);
    auto &encoder_ptr = handler->encoder;
    auto &image_binary = output_it->image_binary;

    size_t max_buffer_size = encoder_ptr->calMaxEncodedDataSize(shape);

    image_binary.resize(max_buffer_size);
    unsigned char * binary_data = (unsigned char *) image_binary.data();
    unsigned char * input_data = (unsigned char*) (input_nd->data);;


    size_t actual_jpeg_length = max_buffer_size;

    encoder_ptr->createState(cu_stream);
    MXCHECK(encoder_ptr->encode(input_data, op_->input_format_, data_type, shape, cu_stream) == EXIT_SUCCESS)
          << "[ImencodeGPU] failed to start encoding.";

    MXCHECK(encoder_ptr->retrieveToHost(binary_data, actual_jpeg_length, cu_stream) == EXIT_SUCCESS)
          << "[ImencodeGPU] failed to retrieve the encoded image to host.";

    CHECK_CUDA_CALL(cudaEventRecord(output_it->finish_event, cu_stream));
}

void ImageEncodeTask::RunImpl() {
  List::iterator input_it = input_it_;
  std::vector<EncodeTaskOutput>::iterator output_it = output_it_;
  auto handler = (op_->get_handler_pool())->borrow();
  if (!no_throw_) {
    for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
      encode(input_it, output_it, handler);
    }
  } else {
    for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
      try {
        encode(input_it, output_it, handler);
      } catch (...) {
        output_it->success = false;
    }
  }
}
}

std::vector<internal::IRunnablePtr> ImageEncodeTask::build_tasks(
    VisionImencodeOpGPU* op,
    List::iterator input_first,
    std::vector<EncodeTaskOutput>::iterator output_first,
    int len,
    int thread_num,
    bool no_throw) {
  std::vector<internal::IRunnablePtr> ret;
  if (len <= thread_num) {
    ret.reserve(len);
    for (int i = 0; i < len; ++i) {
      ret.emplace_back(
          std::make_shared<ImageEncodeTask>(op, input_first + i, output_first + i, 1, no_throw));
    }
    return ret;
  }

  ret.reserve(thread_num);
  int step = len / thread_num;
  int remainder = len % thread_num;
  for (int i = 0; i < remainder; ++i) {
    ret.emplace_back(
        std::make_shared<ImageEncodeTask>(op, input_first, output_first, step + 1, no_throw));
    input_first += step + 1;
    output_first += step + 1;
  }
  for (int i = remainder; i < thread_num; ++i) {
    ret.emplace_back(
        std::make_shared<ImageEncodeTask>(op, input_first, output_first, step, no_throw));
    input_first += step;
    output_first += step;
  }
  return ret;
}






}  // namespace


VisionImencodeOpGPU::VisionImencodeOpGPU(const Any& session_info,
                                         const unicode_view& in_fmt,
                                         const int quality, 
                                         const bool optimized_Huffman,
                                         int pool_size)
    : VisionBaseOpGPU(session_info) {
  MXCHECK(pool_size > 1)
      << "[VisionImencodeOpGPU] pool size must be greater then one and power of 2";
  MXCHECK_EQ((pool_size & (pool_size - 1)), 0)
      << "[VisionImencodeOpGPU] pool size must be greater then one and power of 2";
  local_thread_pool_ =
      new ::matxscript::runtime::internal::LockBasedThreadPool(pool_size, "ImencodeThreadPool");
  pool_size_ = pool_size;
  cv::setNumThreads(0);
  nvjpeg_handle = cuda_op::Encoder::createHandler();
  nvjpeg_param = cuda_op::Encoder::createParameter(nvjpeg_handle, getStream(), quality, optimized_Huffman);
  cuda_op::DataShape dummy_shape;
  dummy_shape.N = 1;
  dummy_shape.C = 3;
  dummy_shape.H = 1024;
  dummy_shape.W = 1024;
  input_format_ = parse_fmt(in_fmt);
  std::vector<std::unique_ptr<EncoderHandlerImpl>> handlers;
  handlers.reserve(pool_size_);
  for (int i = 0; i < pool_size_; ++i) {
    handlers.push_back(std::move(EncoderHandlerImpl::build(nvjpeg_handle, nvjpeg_param, dummy_shape, device_id_)));
  }
  handler_pool_ = std::make_shared<HandlerPool>(std::move(handlers));
}

RTValue VisionImencodeOpGPU::process(const List& arg_images, List* flags) {
  // prepare input & output
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());
  if (images.size() == 0) {
    return List();
  }
  std::vector<EncodeTaskOutput> outputs(images.size());
  bool no_throw = (flags != nullptr);
  auto tasks = ImageEncodeTask::build_tasks(
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
    ret.push_back(std::move(output.image_binary));
  }
  return ret;
}

nvjpegInputFormat_t VisionImencodeOpGPU::parse_fmt(const unicode_view& fmt) {
  if (fmt == U"RGB") {
    return NVJPEG_INPUT_RGBI;
  }
  if (fmt == U"BGR") {
    return NVJPEG_INPUT_BGRI;
  }
  MXTHROW << "Image Encode: output format [" << fmt << "] is invalid, please check carefully.";
  return NVJPEG_INPUT_RGBI ;
}

MATX_REGISTER_NATIVE_OBJECT(VisionImencodeOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 5) << "[VsionImencodeOpGPU] Expect 5 arguments but get "
                                 << args.size();

      return std::make_shared<VisionImencodeOpGPU>(
          args[4], args[0].As<unicode_view>(), args[1].As<int>(), args[2].As<bool>(), args[3].As<int>());
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 1) << "[VsionImencodeOpGPU] Expect 1 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionImencodeOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(), nullptr);
    });

using VisionImencodeNoExceptionOpGPU = VisionImencodeOpGPU;
MATX_REGISTER_NATIVE_OBJECT(VisionImencodeNoExceptionOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 5) << "[VsionImencodeOpGPU] Expect 5 arguments but get "
                                 << args.size();
      return std::make_shared<VisionImencodeOpGPU>(
          args[4], args[0].As<unicode_view>(), args[1].As<int>(), args[2].As<bool>(), args[3].As<int>());
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 1) << "[VisionImencodeOpGPU] Expect 1 arguments but get "
                                 << args.size();
      List flags;
      auto ret = reinterpret_cast<VisionImencodeOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(), &flags);
      return Tuple({ret, flags});
    });
















}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision