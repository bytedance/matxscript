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
#include "opencv_util.h"

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/dlpack.h>
#include <matxscript/runtime/logging.h>
#include <cstdint>
#include "utils/type_helper.h"

namespace byted_matx_vision {
namespace ops {

using matxscript::runtime::DataType;
using matxscript::runtime::DeviceAPI;
using matxscript::runtime::List;
using matxscript::runtime::NDArray;
using matxscript::runtime::unicode_view;

// only support NDArray in HWC format
cv::Mat NDArrayToOpencvMat(const NDArray& ndarray) {
  List shape = ndarray.ShapeList();
  auto* data_ref = ndarray.operator->();
  int64_t height = shape[0].As<int64_t>();
  int64_t width = shape[1].As<int64_t>();
  int64_t dim = shape.size();

  int channels = 0;
  MXCHECK(dim > 1 && dim < 4) << "NDArray Dims must large than 1, and less than 4, but get dim: "
                              << dim;
  if (dim == 2) {
    channels = 1;
  } else {
    channels = shape[2].As<int>();
  }
  int opencv_type = DLDataTypeToOpencvType(ndarray.DataType(), channels);
  if (!ndarray.IsContiguous()) {
    MXLOG(FATAL) << "Don't implements not contiguous NDArray to OpencvMat!";
  }
  cv::Mat matSrc = cv::Mat(cv::Size(width, height), opencv_type, (void*)data_ref->data);
  return matSrc;
}

NDArray OpencvMatToNDArray(const cv::Mat& mat,
                           DLDevice ctx,
                           MATXScriptStreamHandle stream,
                           bool sync) {
  int opencv_depth = mat.depth();
  int channels = mat.channels();
  cv::Size orgsize = mat.size();
  int width = static_cast<int64_t>(orgsize.width);
  int height = static_cast<int64_t>(orgsize.height);

  std::vector<int64_t> ndarry_shape;
  if (channels == 1) {
    ndarry_shape = {height, width};
  } else {
    ndarry_shape = {height, width, channels};
  }

  DataType dtype(OpencvDepthToDLDataType(opencv_depth));
  NDArray dst_arr = NDArray::Empty(ndarry_shape, dtype, ctx);
  DLTensor to;
  to.data = const_cast<void*>(dst_arr.RawData());
  to.shape = ndarry_shape.data();
  to.ndim = ndarry_shape.size();
  to.device = ctx;
  to.strides = nullptr;
  to.byte_offset = 0;
  to.dtype = dtype;

  DLTensor from;
  from.shape = ndarry_shape.data();
  from.ndim = ndarry_shape.size();
  from.device = DLDevice{kDLCPU, 0};
  from.strides = nullptr;
  from.byte_offset = 0;
  from.dtype = dtype;
  if (mat.isContinuous()) {
    from.data = (void*)(mat.data);
    NDArray::CopyFromTo(&from, &to, stream);
    if (sync) {
      DeviceAPI::Get(ctx)->StreamSync(ctx, stream);
    }
    return dst_arr;
  } else {
    cv::Mat clone_mat = mat.clone();
    from.data = (void*)(clone_mat.data);
    NDArray::CopyFromTo(&from, &to, stream);
    if (sync) {
      DeviceAPI::Get(ctx)->StreamSync(ctx, stream);
    }
    return dst_arr;
  }
}

int UnicodeToOpencvInterp(unicode_view opencv_interp) {
  std::unordered_map<unicode_view, cv::InterpolationFlags> cv_interpolation_map = {
      {U"INTER_LINEAR", cv::INTER_LINEAR},
      {U"INTER_NEAREST", cv::INTER_NEAREST},
      {U"INTER_CUBIC", cv::INTER_CUBIC},
      {U"INTER_AREA", cv::INTER_AREA},
      {U"INTER_LANCZOS4", cv::INTER_LANCZOS4},
      {U"INTER_LINEAR_EXACT", cv::INTER_LINEAR_EXACT},
      {U"INTER_MAX", cv::INTER_MAX},
      {U"WARP_FILL_OUTLIERS", cv::WARP_FILL_OUTLIERS},
      {U"WARP_INVERSE_MAP", cv::WARP_INVERSE_MAP},
  };
  // all pillow interp flags are less than 0
  // the actual value would be -v-1, v is the value of the map
  std::unordered_map<unicode_view, int> pillow_interpolation_map = {{U"PILLOW_INTER_LINEAR", -3},
                                                                    {U"PILLOW_INTER_NEAREST", -1},
                                                                    {U"PILLOW_INTER_CUBIC", -4},
                                                                    {U"PILLOW_INTER_LANCZOS4", -2}};

  int cv_interpolation_flags{-1};
  auto it = cv_interpolation_map.find(opencv_interp);
  if (it != cv_interpolation_map.end()) {
    cv_interpolation_flags = it->second;
  } else {  // check if the input interp flag is a pillow one
    auto pit = pillow_interpolation_map.find(opencv_interp);
    if (pit != pillow_interpolation_map.end()) {
      cv_interpolation_flags = pit->second;
    } else {
      MXLOG(FATAL) << "cv_interpolation_flags [" << opencv_interp
                   << "] is invalid, please check carefully.";
    }
  }
  return cv_interpolation_flags;
}

int UnicodeToOpencvColorCode(unicode_view color_code) {
  std::unordered_map<unicode_view, cv::ColorConversionCodes> cv_color_convertion_map = {
      {U"COLOR_BGR2BGRA", cv::COLOR_BGR2BGRA},
      {U"COLOR_RGB2RGBA", cv::COLOR_RGB2RGBA},
      {U"COLOR_BGRA2BGR", cv::COLOR_BGRA2BGR},
      {U"COLOR_RGBA2RGB", cv::COLOR_RGBA2RGB},
      {U"COLOR_BGR2RGBA", cv::COLOR_BGR2RGBA},
      {U"COLOR_RGB2BGRA", cv::COLOR_RGB2BGRA},
      {U"COLOR_RGBA2BGR", cv::COLOR_RGBA2BGR},
      {U"COLOR_BGRA2RGB", cv::COLOR_BGRA2RGB},
      {U"COLOR_BGR2RGB", cv::COLOR_BGR2RGB},
      {U"COLOR_RGB2BGR", cv::COLOR_RGB2BGR},
      {U"COLOR_BGRA2RGBA", cv::COLOR_BGRA2RGBA},
      {U"COLOR_RGBA2BGRA", cv::COLOR_RGBA2BGRA},
      {U"COLOR_BGR2GRAY", cv::COLOR_BGR2GRAY},
      {U"COLOR_RGB2GRAY", cv::COLOR_RGB2GRAY},
      {U"COLOR_GRAY2BGR", cv::COLOR_GRAY2BGR},
      {U"COLOR_GRAY2RGB", cv::COLOR_GRAY2RGB},
      {U"COLOR_GRAY2BGRA", cv::COLOR_GRAY2BGRA},
      {U"COLOR_GRAY2RGBA", cv::COLOR_GRAY2RGBA},
      {U"COLOR_BGRA2GRAY", cv::COLOR_BGRA2GRAY},
      {U"COLOR_RGBA2GRAY", cv::COLOR_RGBA2GRAY},
      {U"COLOR_BGR2BGR565", cv::COLOR_BGR2BGR565},
      {U"COLOR_RGB2BGR565", cv::COLOR_RGB2BGR565},
      {U"COLOR_BGR5652BGR", cv::COLOR_BGR5652BGR},
      {U"COLOR_BGR5652RGB", cv::COLOR_BGR5652RGB},
      {U"COLOR_BGRA2BGR565", cv::COLOR_BGRA2BGR565},
      {U"COLOR_RGBA2BGR565", cv::COLOR_RGBA2BGR565},
      {U"COLOR_BGR5652BGRA", cv::COLOR_BGR5652BGRA},
      {U"COLOR_BGR5652RGBA", cv::COLOR_BGR5652RGBA},
      {U"COLOR_GRAY2BGR565", cv::COLOR_GRAY2BGR565},
      {U"COLOR_BGR5652GRAY", cv::COLOR_BGR5652GRAY},
      {U"COLOR_BGR2BGR555", cv::COLOR_BGR2BGR555},
      {U"COLOR_RGB2BGR555", cv::COLOR_RGB2BGR555},
      {U"COLOR_BGR5552BGR", cv::COLOR_BGR5552BGR},
      {U"COLOR_BGR5552RGB", cv::COLOR_BGR5552RGB},
      {U"COLOR_BGRA2BGR555", cv::COLOR_BGRA2BGR555},
      {U"COLOR_RGBA2BGR555", cv::COLOR_RGBA2BGR555},
      {U"COLOR_BGR5552BGRA", cv::COLOR_BGR5552BGRA},
      {U"COLOR_BGR5552RGBA", cv::COLOR_BGR5552RGBA},
      {U"COLOR_GRAY2BGR555", cv::COLOR_GRAY2BGR555},
      {U"COLOR_BGR5552GRAY", cv::COLOR_BGR5552GRAY},
      {U"COLOR_BGR2XYZ", cv::COLOR_BGR2XYZ},
      {U"COLOR_RGB2XYZ", cv::COLOR_RGB2XYZ},
      {U"COLOR_XYZ2BGR", cv::COLOR_XYZ2BGR},
      {U"COLOR_XYZ2RGB", cv::COLOR_XYZ2RGB},
      {U"COLOR_BGR2YCrCb", cv::COLOR_BGR2YCrCb},
      {U"COLOR_RGB2YCrCb", cv::COLOR_RGB2YCrCb},
      {U"COLOR_YCrCb2BGR", cv::COLOR_YCrCb2BGR},
      {U"COLOR_YCrCb2RGB", cv::COLOR_YCrCb2RGB},
      {U"COLOR_BGR2HSV", cv::COLOR_BGR2HSV},
      {U"COLOR_RGB2HSV", cv::COLOR_RGB2HSV},
      {U"COLOR_BGR2Lab", cv::COLOR_BGR2Lab},
      {U"COLOR_RGB2Lab", cv::COLOR_RGB2Lab},
      {U"COLOR_BGR2Luv", cv::COLOR_BGR2Luv},
      {U"COLOR_RGB2Luv", cv::COLOR_RGB2Luv},
      {U"COLOR_BGR2HLS", cv::COLOR_BGR2HLS},
      {U"COLOR_RGB2HLS", cv::COLOR_RGB2HLS},
      {U"COLOR_HSV2BGR", cv::COLOR_HSV2BGR},
      {U"COLOR_HSV2RGB", cv::COLOR_HSV2RGB},
      {U"COLOR_Lab2BGR", cv::COLOR_Lab2BGR},
      {U"COLOR_Lab2RGB", cv::COLOR_Lab2RGB},
      {U"COLOR_Luv2BGR", cv::COLOR_Luv2BGR},
      {U"COLOR_Luv2RGB", cv::COLOR_Luv2RGB},
      {U"COLOR_HLS2BGR", cv::COLOR_HLS2BGR},
      {U"COLOR_HLS2RGB", cv::COLOR_HLS2RGB},
      {U"COLOR_BGR2HSV_FULL", cv::COLOR_BGR2HSV_FULL},
      {U"COLOR_RGB2HSV_FULL", cv::COLOR_RGB2HSV_FULL},
      {U"COLOR_BGR2HLS_FULL", cv::COLOR_BGR2HLS_FULL},
      {U"COLOR_RGB2HLS_FULL", cv::COLOR_RGB2HLS_FULL},
      {U"COLOR_HSV2BGR_FULL", cv::COLOR_HSV2BGR_FULL},
      {U"COLOR_HSV2RGB_FULL", cv::COLOR_HSV2RGB_FULL},
      {U"COLOR_HLS2BGR_FULL", cv::COLOR_HLS2BGR_FULL},
      {U"COLOR_HLS2RGB_FULL", cv::COLOR_HLS2RGB_FULL},
      {U"COLOR_LBGR2Lab", cv::COLOR_LBGR2Lab},
      {U"COLOR_LRGB2Lab", cv::COLOR_LRGB2Lab},
      {U"COLOR_LBGR2Luv", cv::COLOR_LBGR2Luv},
      {U"COLOR_LRGB2Luv", cv::COLOR_LRGB2Luv},
      {U"COLOR_Lab2LBGR", cv::COLOR_Lab2LBGR},
      {U"COLOR_Lab2LRGB", cv::COLOR_Lab2LRGB},
      {U"COLOR_Luv2LBGR", cv::COLOR_Luv2LBGR},
      {U"COLOR_Luv2LRGB", cv::COLOR_Luv2LRGB},
      {U"COLOR_BGR2YUV", cv::COLOR_BGR2YUV},
      {U"COLOR_RGB2YUV", cv::COLOR_RGB2YUV},
      {U"COLOR_YUV2BGR", cv::COLOR_YUV2BGR},
      {U"COLOR_YUV2RGB", cv::COLOR_YUV2RGB},
      {U"COLOR_YUV2RGB_NV12", cv::COLOR_YUV2RGB_NV12},
      {U"COLOR_YUV2BGR_NV12", cv::COLOR_YUV2BGR_NV12},
      {U"COLOR_YUV2RGB_NV21", cv::COLOR_YUV2RGB_NV21},
      {U"COLOR_YUV2BGR_NV21", cv::COLOR_YUV2BGR_NV21},
      {U"COLOR_YUV420sp2RGB", cv::COLOR_YUV420sp2RGB},
      {U"COLOR_YUV420sp2BGR", cv::COLOR_YUV420sp2BGR},
      {U"COLOR_YUV2RGBA_NV12", cv::COLOR_YUV2RGBA_NV12},
      {U"COLOR_YUV2BGRA_NV12", cv::COLOR_YUV2BGRA_NV12},
      {U"COLOR_YUV2RGBA_NV21", cv::COLOR_YUV2RGBA_NV21},
      {U"COLOR_YUV2BGRA_NV21", cv::COLOR_YUV2BGRA_NV21},
      {U"COLOR_YUV420sp2RGBA", cv::COLOR_YUV420sp2RGBA},
      {U"COLOR_YUV420sp2BGRA", cv::COLOR_YUV420sp2BGRA},
      {U"COLOR_YUV2RGB_YV12", cv::COLOR_YUV2RGB_YV12},
      {U"COLOR_YUV2BGR_YV12", cv::COLOR_YUV2BGR_YV12},
      {U"COLOR_YUV2RGB_IYUV", cv::COLOR_YUV2RGB_IYUV},
      {U"COLOR_YUV2BGR_IYUV", cv::COLOR_YUV2BGR_IYUV},
      {U"COLOR_YUV2RGB_I420", cv::COLOR_YUV2RGB_I420},
      {U"COLOR_YUV2BGR_I420", cv::COLOR_YUV2BGR_I420},
      {U"COLOR_YUV420p2RGB", cv::COLOR_YUV420p2RGB},
      {U"COLOR_YUV420p2BGR", cv::COLOR_YUV420p2BGR},
      {U"COLOR_YUV2RGBA_YV12", cv::COLOR_YUV2RGBA_YV12},
      {U"COLOR_YUV2BGRA_YV12", cv::COLOR_YUV2BGRA_YV12},
      {U"COLOR_YUV2RGBA_IYUV", cv::COLOR_YUV2RGBA_IYUV},
      {U"COLOR_YUV2BGRA_IYUV", cv::COLOR_YUV2BGRA_IYUV},
      {U"COLOR_YUV2RGBA_I420", cv::COLOR_YUV2RGBA_I420},
      {U"COLOR_YUV2BGRA_I420", cv::COLOR_YUV2BGRA_I420},
      {U"COLOR_YUV420p2RGBA", cv::COLOR_YUV420p2RGBA},
      {U"COLOR_YUV420p2BGRA", cv::COLOR_YUV420p2BGRA},
      {U"COLOR_YUV2GRAY_420", cv::COLOR_YUV2GRAY_420},
      {U"COLOR_YUV2GRAY_NV21", cv::COLOR_YUV2GRAY_NV21},
      {U"COLOR_YUV2GRAY_NV12", cv::COLOR_YUV2GRAY_NV12},
      {U"COLOR_YUV2GRAY_YV12", cv::COLOR_YUV2GRAY_YV12},
      {U"COLOR_YUV2GRAY_IYUV", cv::COLOR_YUV2GRAY_IYUV},
      {U"COLOR_YUV2GRAY_I420", cv::COLOR_YUV2GRAY_I420},
      {U"COLOR_YUV420sp2GRAY", cv::COLOR_YUV420sp2GRAY},
      {U"COLOR_YUV420p2GRAY", cv::COLOR_YUV420p2GRAY},
      {U"COLOR_YUV2RGB_UYVY", cv::COLOR_YUV2RGB_UYVY},
      {U"COLOR_YUV2BGR_UYVY", cv::COLOR_YUV2BGR_UYVY},
      {U"COLOR_YUV2RGB_Y422", cv::COLOR_YUV2RGB_Y422},
      {U"COLOR_YUV2BGR_Y422", cv::COLOR_YUV2BGR_Y422},
      {U"COLOR_YUV2RGB_UYNV", cv::COLOR_YUV2RGB_UYNV},
      {U"COLOR_YUV2BGR_UYNV", cv::COLOR_YUV2BGR_UYNV},
      {U"COLOR_YUV2RGBA_UYVY", cv::COLOR_YUV2RGBA_UYVY},
      {U"COLOR_YUV2BGRA_UYVY", cv::COLOR_YUV2BGRA_UYVY},
      {U"COLOR_YUV2RGBA_Y422", cv::COLOR_YUV2RGBA_Y422},
      {U"COLOR_YUV2BGRA_Y422", cv::COLOR_YUV2BGRA_Y422},
      {U"COLOR_YUV2RGBA_UYNV", cv::COLOR_YUV2RGBA_UYNV},
      {U"COLOR_YUV2BGRA_UYNV", cv::COLOR_YUV2BGRA_UYNV},
      {U"COLOR_YUV2RGB_YUY2", cv::COLOR_YUV2RGB_YUY2},
      {U"COLOR_YUV2BGR_YUY2", cv::COLOR_YUV2BGR_YUY2},
      {U"COLOR_YUV2RGB_YVYU", cv::COLOR_YUV2RGB_YVYU},
      {U"COLOR_YUV2BGR_YVYU", cv::COLOR_YUV2BGR_YVYU},
      {U"COLOR_YUV2RGB_YUYV", cv::COLOR_YUV2RGB_YUYV},
      {U"COLOR_YUV2BGR_YUYV", cv::COLOR_YUV2BGR_YUYV},
      {U"COLOR_YUV2RGB_YUNV", cv::COLOR_YUV2RGB_YUNV},
      {U"COLOR_YUV2BGR_YUNV", cv::COLOR_YUV2BGR_YUNV},
      {U"COLOR_YUV2RGBA_YUY2", cv::COLOR_YUV2RGBA_YUY2},
      {U"COLOR_YUV2BGRA_YUY2", cv::COLOR_YUV2BGRA_YUY2},
      {U"COLOR_YUV2RGBA_YVYU", cv::COLOR_YUV2RGBA_YVYU},
      {U"COLOR_YUV2BGRA_YVYU", cv::COLOR_YUV2BGRA_YVYU},
      {U"COLOR_YUV2RGBA_YUYV", cv::COLOR_YUV2RGBA_YUYV},
      {U"COLOR_YUV2BGRA_YUYV", cv::COLOR_YUV2BGRA_YUYV},
      {U"COLOR_YUV2RGBA_YUNV", cv::COLOR_YUV2RGBA_YUNV},
      {U"COLOR_YUV2BGRA_YUNV", cv::COLOR_YUV2BGRA_YUNV},
      {U"COLOR_YUV2GRAY_UYVY", cv::COLOR_YUV2GRAY_UYVY},
      {U"COLOR_YUV2GRAY_YUY2", cv::COLOR_YUV2GRAY_YUY2},
      {U"COLOR_YUV2GRAY_Y422", cv::COLOR_YUV2GRAY_Y422},
      {U"COLOR_YUV2GRAY_UYNV", cv::COLOR_YUV2GRAY_UYNV},
      {U"COLOR_YUV2GRAY_YVYU", cv::COLOR_YUV2GRAY_YVYU},
      {U"COLOR_YUV2GRAY_YUYV", cv::COLOR_YUV2GRAY_YUYV},
      {U"COLOR_YUV2GRAY_YUNV", cv::COLOR_YUV2GRAY_YUNV},
      {U"COLOR_RGBA2mRGBA", cv::COLOR_RGBA2mRGBA},
      {U"COLOR_mRGBA2RGBA", cv::COLOR_mRGBA2RGBA},
      {U"COLOR_RGB2YUV_I420", cv::COLOR_RGB2YUV_I420},
      {U"COLOR_BGR2YUV_I420", cv::COLOR_BGR2YUV_I420},
      {U"COLOR_RGB2YUV_IYUV", cv::COLOR_RGB2YUV_IYUV},
      {U"COLOR_BGR2YUV_IYUV", cv::COLOR_BGR2YUV_IYUV},
      {U"COLOR_RGBA2YUV_I420", cv::COLOR_RGBA2YUV_I420},
      {U"COLOR_BGRA2YUV_I420", cv::COLOR_BGRA2YUV_I420},
      {U"COLOR_RGBA2YUV_IYUV", cv::COLOR_RGBA2YUV_IYUV},
      {U"COLOR_BGRA2YUV_IYUV", cv::COLOR_BGRA2YUV_IYUV},
      {U"COLOR_RGB2YUV_YV12", cv::COLOR_RGB2YUV_YV12},
      {U"COLOR_BGR2YUV_YV12", cv::COLOR_BGR2YUV_YV12},
      {U"COLOR_RGBA2YUV_YV12", cv::COLOR_RGBA2YUV_YV12},
      {U"COLOR_BGRA2YUV_YV12", cv::COLOR_BGRA2YUV_YV12},
      {U"COLOR_BayerBG2BGR", cv::COLOR_BayerBG2BGR},
      {U"COLOR_BayerGB2BGR", cv::COLOR_BayerGB2BGR},
      {U"COLOR_BayerRG2BGR", cv::COLOR_BayerRG2BGR},
      {U"COLOR_BayerGR2BGR", cv::COLOR_BayerGR2BGR},
      {U"COLOR_BayerBG2RGB", cv::COLOR_BayerBG2RGB},
      {U"COLOR_BayerGB2RGB", cv::COLOR_BayerGB2RGB},
      {U"COLOR_BayerRG2RGB", cv::COLOR_BayerRG2RGB},
      {U"COLOR_BayerGR2RGB", cv::COLOR_BayerGR2RGB},
      {U"COLOR_BayerBG2GRAY", cv::COLOR_BayerBG2GRAY},
      {U"COLOR_BayerGB2GRAY", cv::COLOR_BayerGB2GRAY},
      {U"COLOR_BayerRG2GRAY", cv::COLOR_BayerRG2GRAY},
      {U"COLOR_BayerGR2GRAY", cv::COLOR_BayerGR2GRAY},
      {U"COLOR_BayerBG2BGR_VNG", cv::COLOR_BayerBG2BGR_VNG},
      {U"COLOR_BayerGB2BGR_VNG", cv::COLOR_BayerGB2BGR_VNG},
      {U"COLOR_BayerRG2BGR_VNG", cv::COLOR_BayerRG2BGR_VNG},
      {U"COLOR_BayerGR2BGR_VNG", cv::COLOR_BayerGR2BGR_VNG},
      {U"COLOR_BayerBG2RGB_VNG", cv::COLOR_BayerBG2RGB_VNG},
      {U"COLOR_BayerGB2RGB_VNG", cv::COLOR_BayerGB2RGB_VNG},
      {U"COLOR_BayerRG2RGB_VNG", cv::COLOR_BayerRG2RGB_VNG},
      {U"COLOR_BayerGR2RGB_VNG", cv::COLOR_BayerGR2RGB_VNG},
      {U"COLOR_BayerBG2BGR_EA", cv::COLOR_BayerBG2BGR_EA},
      {U"COLOR_BayerGB2BGR_EA", cv::COLOR_BayerGB2BGR_EA},
      {U"COLOR_BayerRG2BGR_EA", cv::COLOR_BayerRG2BGR_EA},
      {U"COLOR_BayerGR2BGR_EA", cv::COLOR_BayerGR2BGR_EA},
      {U"COLOR_BayerBG2RGB_EA", cv::COLOR_BayerBG2RGB_EA},
      {U"COLOR_BayerGB2RGB_EA", cv::COLOR_BayerGB2RGB_EA},
      {U"COLOR_BayerRG2RGB_EA", cv::COLOR_BayerRG2RGB_EA},
      {U"COLOR_BayerGR2RGB_EA", cv::COLOR_BayerGR2RGB_EA},
      {U"COLOR_BayerBG2BGRA", cv::COLOR_BayerBG2BGRA},
      {U"COLOR_BayerGB2BGRA", cv::COLOR_BayerGB2BGRA},
      {U"COLOR_BayerRG2BGRA", cv::COLOR_BayerRG2BGRA},
      {U"COLOR_BayerGR2BGRA", cv::COLOR_BayerGR2BGRA},
      {U"COLOR_BayerBG2RGBA", cv::COLOR_BayerBG2RGBA},
      {U"COLOR_BayerGB2RGBA", cv::COLOR_BayerGB2RGBA},
      {U"COLOR_BayerRG2RGBA", cv::COLOR_BayerRG2RGBA},
      {U"COLOR_BayerGR2RGBA", cv::COLOR_BayerGR2RGBA},
      {U"COLOR_COLORCVT_MAX", cv::COLOR_COLORCVT_MAX}};
  int cv_color_code{-1};
  auto it = cv_color_convertion_map.find(color_code);
  if (it != cv_color_convertion_map.end()) {
    cv_color_code = it->second;
  } else {
    MXLOG(FATAL) << "code [" << color_code << "] is invalid, please check carefully.";
  }
  return cv_color_code;
}

}  // namespace ops
}  // namespace byted_matx_vision
