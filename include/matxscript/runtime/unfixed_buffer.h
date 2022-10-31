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

#include "runtime_port.h"

#include <cstdlib>
#include <cstring>

namespace matxscript {
namespace runtime {

template <typename T, int32_t MEAN_SIZE = 256>
class UnfixedBuffer {
 public:
  UnfixedBuffer() = default;
  ~UnfixedBuffer() noexcept {
    if (buf_large_base_) {
      free(buf_large_base_);
      buf_large_base_ = nullptr;
      buf_large_ele_ = nullptr;
    }
  }

  UnfixedBuffer(const UnfixedBuffer& other) = delete;
  UnfixedBuffer& operator=(const UnfixedBuffer& other) = delete;
  UnfixedBuffer(UnfixedBuffer&& other) = delete;
  UnfixedBuffer& operator=(UnfixedBuffer&& other) = delete;

  /**
   *
   *
   * @param len
   * @return
   */
  T* Data(int32_t len, T* old = nullptr, int32_t old_len = 0) noexcept {
    if (len > MEAN_SIZE) {
      if (len < large_buf_size_ && buf_large_ele_ && buf_large_base_) {
        if (old && old_len > 0 && old != buf_large_ele_) {
          memmove(buf_large_ele_,
                  old,
                  large_buf_size_ > old_len ? old_len * sizeof(T) : large_buf_size_ * sizeof(T));
        }
        return buf_large_ele_;
      } else {
        auto buf_base_tmp = malloc(len * sizeof(T) + MATXSCRIPT_MEMORY_ALIGNMENT - 1);
        if (!buf_base_tmp) {
          return nullptr;
        }
        auto buf_ele_tmp = reinterpret_cast<T*>(
            matxscript_memory_align_ptr(buf_base_tmp, MATXSCRIPT_MEMORY_ALIGNMENT));
        if (old && old_len > 0) {
          memmove(buf_ele_tmp, old, len > old_len ? old_len * sizeof(T) : len * sizeof(T));
        }
        if (buf_large_base_) {
          free(buf_large_base_);
        }
        buf_large_base_ = buf_base_tmp;
        buf_large_ele_ = buf_ele_tmp;
        large_buf_size_ = len;
        return buf_large_ele_;
      }
    } else {
      if (old && old_len > 0 && old != buf_small_) {
        memmove(buf_small_, old, MEAN_SIZE > old_len ? old_len * sizeof(T) : MEAN_SIZE * sizeof(T));
      }
      return buf_small_;
    }
  }

 private:
  T buf_small_[MEAN_SIZE];
  void* buf_large_base_ = nullptr;
  T* buf_large_ele_ = nullptr;
  int32_t large_buf_size_ = 0;
};

}  // namespace runtime
}  // namespace matxscript
