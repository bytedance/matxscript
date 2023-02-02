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
#include "utils/pad_types.h"

namespace byted_matx_vision {
namespace ops {

int UnicodePadTypesToCVBorderTypes(matxscript::runtime::unicode_view pad_type) {
  CvBorderTypesMap_t cv_border_types_map = {{U"BORDER_CONSTANT", cv::BORDER_CONSTANT},
                                            {U"BORDER_REPLICATE", cv::BORDER_REPLICATE},
                                            {U"BORDER_REFLECT", cv::BORDER_REFLECT},
                                            {U"BORDER_WRAP", cv::BORDER_WRAP},
                                            {U"BORDER_REFLECT_101", cv::BORDER_REFLECT_101},
                                            {U"BORDER_TRANSPARENT", cv::BORDER_TRANSPARENT},
                                            {U"BORDER_REFLECT101", cv::BORDER_REFLECT101},
                                            {U"BORDER_DEFAULT", cv::BORDER_DEFAULT},
                                            {U"BORDER_ISOLATED", cv::BORDER_ISOLATED}};
  int cv_border_type{-1};
  auto it = cv_border_types_map.find(pad_type);
  if (it != cv_border_types_map.end()) {
    cv_border_type = it->second;
  } else {
    MXCHECK(false) << "cv_border_type [" << pad_type << "] is invalidate, please check carefully.";
  }
  return cv_border_type;
}

}  // namespace ops
}  // namespace byted_matx_vision