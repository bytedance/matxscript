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
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/string_util.h>

namespace matxscript {
namespace runtime {

namespace internal {
inline int cuda_device_offset(int gpu_ordinal) {
  if (gpu_ordinal < 0) {
    return gpu_ordinal;
  }
  // use `index` instead of `device.oridinal` for pytorch engine
  // e.g. if CUDA_VISIBLE_DEVICES=5,6,7 here should be 0,1,2
  char const* tmp = getenv("CUDA_VISIBLE_DEVICES");
  if (tmp == nullptr) {
    // env not set, use all gpus
    return gpu_ordinal;
  }

  std::string env_val(tmp);
  std::vector<std::string> gpu_list = StringUtil::Split(env_val, ",");
  auto it = std::find(gpu_list.begin(), gpu_list.end(), std::to_string(gpu_ordinal));
  MXCHECK(it != gpu_list.end()) << "gpu_ordinal:" << gpu_ordinal
                                << " not found in CUDA_VISIBLE_DEVICES:" << env_val;
  return std::distance(gpu_list.begin(), it);
}

inline int get_tf_inter_op_threads() {
  char const* tmp = getenv("TF_INTER_OP_THREADS");
  if (tmp == nullptr) {
    // env not set, use default 1
    return 1;
  }
  try {
    return std::stoi(std::string(tmp));
  } catch (...) {
    return 1;
  }
}

inline int get_tf_intra_op_threads() {
  char const* tmp = getenv("TF_INTRA_OP_THREADS");
  if (tmp == nullptr) {
    // env not set, use default 1
    return 1;
  }
  try {
    return std::stoi(std::string(tmp));
  } catch (...) {
    return 1;
  }
}

inline void ListToStdVector_String(const List& list, std::vector<std::string>& result) {
  result.reserve(list.size());
  for (auto& item : list) {
    switch (item.type_code()) {
      case TypeIndex::kRuntimeString: {
        auto item_unwrapped = item.As<String>();
        result.push_back(std::move(item_unwrapped));
      } break;
      case TypeIndex::kRuntimeUnicode: {
        auto item_unwrapped = item.As<Unicode>().encode();
        result.push_back(std::move(item_unwrapped));
      } break;
      default: {
        /* not compatible type */
        MXCHECK(false) << "input type error, \n"
                       << "optional: str, \n"
                       << "but receive type : " << item.type_name();
      } break;
    }
  }
}

}  // namespace internal

}  // namespace runtime
}  // namespace matxscript
