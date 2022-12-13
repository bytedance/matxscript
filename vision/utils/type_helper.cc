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
#include "type_helper.h"

#include <opencv2/core/hal/interface.h>

#include <matxscript/runtime/logging.h>

namespace byted_matx_vision {
namespace ops {

using matxscript::runtime::DataType;
using matxscript::runtime::Unicode;
using matxscript::runtime::unicode_view;

DLDataType OpencvDepthToDLDataType(int opencv_depth) {
  DLDataType t;
  t.lanes = 1;
  switch (opencv_depth) {
    // CV_8U
    case CV_8U: {
      t.bits = 8;
      t.code = kDLUInt;
      return t;
    } break;
    // CV_8S
    case CV_8S: {
      t.bits = 8;
      t.code = kDLInt;
      return t;
    } break;
    // CV_16U
    case CV_16U: {
      t.bits = 16;
      t.code = kDLUInt;
      return t;
    } break;
    // CV_16S
    case CV_16S: {
      t.bits = 16;
      t.code = kDLInt;
      return t;
    } break;
    // CV_32S
    case CV_32S: {
      t.bits = 32;
      t.code = kDLInt;
      return t;
    } break;
    // CV_32F
    case CV_32F: {
      t.bits = 32;
      t.code = kDLFloat;
      return t;
    } break;
    // CV_64F
    case CV_64F: {
      t.bits = 64;
      t.code = kDLFloat;
      return t;
    } break;
  }
  MXLOG(FATAL) << "unknown type " << opencv_depth;
  return t;
}

int DLDataTypeToOpencvDepth(DataType dtype) {
  uint8_t type_code = dtype.code();
  uint8_t bits = dtype.bits();

  switch (type_code) {
    case kDLInt: {
      // CV_8S
      if (bits == 8) {
        return CV_8S;
        // CV_16S
      } else if (bits == 16) {
        return CV_16S;
        // CV_32S
      } else if (bits == 32) {
        return CV_32S;
      } else {
        MXLOG(FATAL) << "unknown type " << type_code << " bits: " << bits;
        return CV_USRTYPE1;
      }
    } break;
    case kDLUInt: {
      // CV_8U
      if (bits == 8) {
        return CV_8U;
        // CV_16U
      } else if (bits == 16) {
        return CV_16U;
      } else {
        MXLOG(FATAL) << "unknown type " << type_code << " bits: " << bits;
        return CV_USRTYPE1;
      }

    } break;
    case kDLFloat: {
      // CV_32F
      if (bits == 32) {
        return CV_32F;
      } else if (bits == 64) {
        return CV_64F;
      }
    } break;
  }
  MXLOG(FATAL) << "unknown type code " << type_code;
  return CV_USRTYPE1;
}

int DLDataTypeToOpencvType(DataType dtype, int dim) {
  uint8_t type_code = dtype.code();
  uint8_t bits = dtype.bits();

  switch (type_code) {
    case kDLInt: {
      // CV_8S
      if (bits == 8) {
        return CV_MAKETYPE(CV_8S, dim);
        // CV_16S
      } else if (bits == 16) {
        return CV_MAKETYPE(CV_16S, dim);
        // CV_32S
      } else if (bits == 32) {
        return CV_MAKETYPE(CV_32S, dim);
      } else {
        MXLOG(FATAL) << "unknown type " << type_code << " bits: " << bits;
        return CV_MAKETYPE(CV_USRTYPE1, dim);
      }
    } break;
    case kDLUInt: {
      // CV_8U
      if (bits == 8) {
        return CV_MAKETYPE(CV_8U, dim);
        // CV_16U
      } else if (bits == 16) {
        return CV_MAKETYPE(CV_16U, dim);
      } else {
        MXLOG(FATAL) << "unknown type " << type_code << " bits: " << bits;
        return CV_MAKETYPE(CV_USRTYPE1, dim);
      }

    } break;
    case kDLFloat: {
      // CV_32F
      if (bits == 32) {
        return CV_MAKETYPE(CV_32F, dim);
      } else if (bits == 64) {
        return CV_MAKETYPE(CV_64F, dim);
      }
    } break;
  }
  MXLOG(FATAL) << "unknown type_code " << type_code;
  return CV_MAKETYPE(CV_USRTYPE1, dim);
}

int UnicodeTypeToOpencvDepth(unicode_view opencv_depth) {
  using CvDepthMap_t = std::unordered_map<unicode_view, int>;
  CvDepthMap_t cv_depth_type_map = {{U"uint8", CV_8U},
                                    {U"int8", CV_8S},
                                    {U"uint16", CV_16U},
                                    {U"int16", CV_16S},
                                    {U"int32", CV_32S},
                                    {U"float32", CV_32F},
                                    {U"float64", CV_64F}};
  MXCHECK_GT(opencv_depth.size(), 0) << "Unicode type is empty, please check !";
  int cv_depth_type{-1};
  auto it = cv_depth_type_map.find(opencv_depth);
  if (it != cv_depth_type_map.end()) {
    cv_depth_type = it->second;
  } else {
    MXLOG(FATAL) << "opencv_depth_type [" << opencv_depth << "] is invalidate, please check !";
  }
  return cv_depth_type;
}

}  // namespace ops
}  // namespace byted_matx_vision