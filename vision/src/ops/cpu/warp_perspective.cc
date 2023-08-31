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
#include <opencv2/imgproc.hpp>
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/py_args.h"
#include "matxscript/runtime/runtime_value.h"
#include "matxscript/runtime/threadpool/lock_based_thread_pool.h"
#include "ops/base/vision_base_op.h"
#include "utils/ndarray_helper.h"
#include "utils/opencv_util.h"
#include "utils/pad_types.h"
#include "utils/type_helper.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

namespace {

RTValue warp_perspective_func(const NDArray& image,
                              const Tuple& dsize,
                              const List& src_n_pts,
                              const List& dst_n_pts,
                              const int interp,
                              const int pad_type,
                              const cv::Scalar& pad_value) {
  std::vector<cv::Point2f> src_n_pts_vec, dst_n_pts_vec;
  src_n_pts_vec.reserve(src_n_pts.size());
  dst_n_pts_vec.reserve(dst_n_pts.size());
  for (const auto& pt : src_n_pts) {
    const auto& pt_tuple_view = pt.AsObjectView<Tuple>();
    const auto& pt_tuple = pt_tuple_view.data();
    src_n_pts_vec.push_back(cv::Point(pt_tuple[0].As<float>(), pt_tuple[1].As<float>()));
  }
  for (const auto& pt : dst_n_pts) {
    const auto& pt_tuple_view = pt.AsObjectView<Tuple>();
    const auto& pt_tuple = pt_tuple_view.data();
    dst_n_pts_vec.push_back(cv::Point(pt_tuple[0].As<float>(), pt_tuple[1].As<float>()));
  }
  cv::Size cv_dsize(dsize[0].As<int>(), dsize[1].As<int>());
  const auto& p_transform =
      cv::getPerspectiveTransform(std::move(src_n_pts_vec), std::move(dst_n_pts_vec));
  cv::Mat&& image_mat = NDArrayToOpencvMat(image);
  cv::Mat out_image;
  cv::warpPerspective(std::move(image_mat),
                      out_image,
                      std::move(p_transform),
                      std::move(cv_dsize),
                      interp,
                      pad_type,
                      pad_value);
  return OpencvMatToNDArray(std::move(out_image));
}

class WarpPerspectiveTask : public internal::LockBasedRunnable {
 public:
  WarpPerspectiveTask(List::iterator input_first,
                      List::iterator dsize_first,
                      List::iterator pts_first,
                      const int interp,
                      const int pad_type,
                      const cv::Scalar pad_value,
                      List::iterator output_first,
                      int len)
      : input_it_(input_first),
        dsize_first_(dsize_first),
        pts_first_(pts_first),
        interp_(interp),
        pad_type_(pad_type),
        pad_value_(pad_value),
        output_it_(output_first),
        len_(len) {
  }

  static std::vector<internal::IRunnablePtr> build_tasks(List::iterator input_first,
                                                         List::iterator dsize_first,
                                                         List::iterator pts_first,
                                                         const int interp,
                                                         const int pad_type,
                                                         const cv::Scalar pad_value,
                                                         List::iterator output_first,
                                                         int len,
                                                         int thread_num);

 protected:
  void RunImpl() override;

 private:
  List::iterator input_it_;
  List::iterator dsize_first_;
  List::iterator pts_first_;
  const int interp_;
  const int pad_type_;
  const cv::Scalar pad_value_;
  List::iterator output_it_;
  int len_;
};

void WarpPerspectiveTask::RunImpl() {
  for (int i = 0; i < len_; ++i) {
    const auto& img_view = (input_it_ + i)->AsObjectView<NDArray>();
    const auto& img = img_view.data();
    const auto& dsize_view = (dsize_first_ + i)->AsObjectView<Tuple>();
    const auto& dsize = dsize_view.data();
    const auto& pts_view = (pts_first_ + i)->AsObjectView<List>();
    const auto& pts = pts_view.data();
    const auto& src_n_pts_view = pts[0].AsObjectView<List>();
    const auto& src_n_pts = src_n_pts_view.data();
    const auto& dst_n_pts_view = pts[1].AsObjectView<List>();
    const auto& dst_n_pts = dst_n_pts_view.data();
    *(output_it_ + i) =
        warp_perspective_func(img, dsize, src_n_pts, dst_n_pts, interp_, pad_type_, pad_value_);
  }
}

std::vector<internal::IRunnablePtr> WarpPerspectiveTask::build_tasks(List::iterator input_first,
                                                                     List::iterator dsize_first,
                                                                     List::iterator pts_first,
                                                                     const int interp,
                                                                     const int pad_type,
                                                                     const cv::Scalar pad_value,
                                                                     List::iterator output_first,
                                                                     int len,
                                                                     int thread_num) {
  std::vector<internal::IRunnablePtr> ret;
  if (len <= thread_num) {
    ret.reserve(len);
    for (int i = 0; i < len; ++i) {
      ret.emplace_back(std::make_shared<WarpPerspectiveTask>(input_first + i,
                                                             dsize_first + i,
                                                             pts_first + i,
                                                             interp,
                                                             pad_type,
                                                             pad_value,
                                                             output_first + i,
                                                             1));
    }
    return ret;
  }

  ret.reserve(thread_num);
  int step = len / thread_num;
  int remainder = len % thread_num;

  for (int i = 0; i < remainder; ++i) {
    ret.emplace_back(std::make_shared<WarpPerspectiveTask>(
        input_first, dsize_first, pts_first, interp, pad_type, pad_value, output_first, step + 1));
    input_first += step + 1;
    dsize_first += step + 1;
    pts_first += step + 1;
    output_first += step + 1;
  }
  for (int i = remainder; i < thread_num; ++i) {
    ret.emplace_back(std::make_shared<WarpPerspectiveTask>(
        input_first, dsize_first, pts_first, interp, pad_type, pad_value, output_first, step));
    input_first += step;
    dsize_first += step;
    pts_first += step;
    output_first += step;
  }
  return ret;
}

}  // namespace

class VisionWarpPerspectiveOpCPU : VisionBaseOpCPU {
 public:
  VisionWarpPerspectiveOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
    if (thread_pool_ != nullptr) {
      thread_num_ = thread_pool_->GetThreadsNum();
    }
  }
  ~VisionWarpPerspectiveOpCPU() = default;

  RTValue process(const List& images,
                  const List& dsizes,
                  const List& pts,
                  const unicode_view& borderType,
                  const Tuple& borderValue,
                  const unicode_view& interpolation);

 private:
  int thread_num_ = 0;
};

RTValue VisionWarpPerspectiveOpCPU::process(const List& images,
                                            const List& dsizes,
                                            const List& pts,
                                            const unicode_view& borderType,
                                            const Tuple& borderValue,
                                            const unicode_view& interpolation) {
  if (images.size() == 0) {
    return List();
  }
  if (borderValue.size() != 1 && borderValue.size() != 3) {
    MXCHECK(false) << "The shape of border value should either be 1 or be 3.";
  }

  int interp_flags = UnicodeToOpencvInterp(interpolation);
  int cv_border_type = UnicodePadTypesToCVBorderTypes(borderType);
  cv::Scalar scalar_pad_value;
  if (borderValue.size() == 1) {
    scalar_pad_value = cv::Scalar(borderValue[0].As<float>(), 0, 0);
  } else {
    scalar_pad_value = cv::Scalar(
        borderValue[0].As<float>(), borderValue[1].As<float>(), borderValue[2].As<float>());
  }

  List ret(images.size(), None);
  // build tasks

  auto tasks = WarpPerspectiveTask::build_tasks(images.begin(),
                                                dsizes.begin(),
                                                pts.begin(),
                                                interp_flags,
                                                cv_border_type,
                                                scalar_pad_value,
                                                ret.begin(),
                                                images.size(),
                                                thread_num_ + 1);

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

class VisionWarpPerspectiveGeneralOp : public VisionBaseOp {
 public:
  VisionWarpPerspectiveGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionWarpPerspectiveOp") {
  }
  ~VisionWarpPerspectiveGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionWarpPerspectiveOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionWarpPerspectiveOpCPU] Expect 1 arguments but get "
                                << args.size();
      return std::make_shared<VisionWarpPerspectiveOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 7)
          << "[VisionWarpPerspectiveOpCPU][func: process] Expect 7 arguments but get "
          << args.size();
      return reinterpret_cast<VisionWarpPerspectiveOpCPU*>(self)->process(
          args[0].AsObjectView<List>().data(),
          args[1].AsObjectView<List>().data(),
          args[2].AsObjectView<List>().data(),
          args[3].As<unicode_view>(),
          args[4].AsObjectView<Tuple>().data(),
          args[5].As<unicode_view>());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionWarpPerspectiveGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionWarpPerspectiveGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionWarpPerspectiveGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision
