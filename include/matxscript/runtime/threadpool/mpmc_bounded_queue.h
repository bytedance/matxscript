// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * Taken from http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
 *
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

#include <cstddef>
#include <cstdint>

#include <atomic>
#include <memory>

namespace matxscript {
namespace runtime {

template <typename T>
class MPMCBoundedQueue {
 public:
  /**
   * buffer_size must be 2^n
   * @param buffer_size
   */
  MPMCBoundedQueue(size_t buffer_size)
      : buffer_(new cell_t[buffer_size]), buffer_mask_(buffer_size - 1) {
    if (!((buffer_size >= 2) && ((buffer_size & (buffer_size - 1)) == 0))) {
      abort();
    }
    for (size_t i = 0; i != buffer_size; i += 1)
      buffer_[i].sequence_.store(i, std::memory_order_relaxed);
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
  }

  virtual ~MPMCBoundedQueue() {
    delete[] buffer_;
  }

  template <class U>
  bool enqueue(U&& data) {
    cell_t* cell;
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & buffer_mask_];
      size_t seq = cell->sequence_.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)pos;
      if (dif == 0) {
        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
          break;
      } else if (dif < 0)
        return false;
      else
        pos = enqueue_pos_.load(std::memory_order_relaxed);
    }
    cell->data_ = std::forward<U>(data);
    cell->sequence_.store(pos + 1, std::memory_order_release);
    return true;
  }

  template <class U>
  bool try_enqueue(U&& data) {
    cell_t* cell;
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    cell = &buffer_[pos & buffer_mask_];
    size_t seq = cell->sequence_.load(std::memory_order_acquire);
    intptr_t dif = (intptr_t)seq - (intptr_t)pos;
    if (dif == 0) {
      if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        cell->data_ = std::forward<U>(data);
        cell->sequence_.store(pos + 1, std::memory_order_release);
        return true;
      }
    }
    return false;
  }

  // problem: heavy enqueue competition, need better implementation in the future
  template <class U>
  bool enqueue_bulk(U* data, size_t size) {
    cell_t* cell;
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & buffer_mask_];
      size_t seq = cell->sequence_.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)pos;
      if (dif == 0) {
        if (enqueue_pos_.compare_exchange_weak(pos, pos + size, std::memory_order_relaxed))
          break;
      } else if (dif < 0)
        return false;
      else
        pos = enqueue_pos_.load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < size; ++i) {
      cell_t* cell = &buffer_[(pos + i) & buffer_mask_];
      cell->data_ = data[i];
      cell->sequence_.store(pos + 1 + i, std::memory_order_release);
    }
    return true;
  }

  bool dequeue(T& data) {
    cell_t* cell;
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & buffer_mask_];
      size_t seq = cell->sequence_.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
      if (dif == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
          break;
      } else if (dif < 0)
        return false;
      else
        pos = dequeue_pos_.load(std::memory_order_relaxed);
    }
    data = std::move(cell->data_);
    cell->sequence_.store(pos + buffer_mask_ + 1, std::memory_order_release);
    return true;
  }

  bool try_dequeue(T& data) {
    cell_t* cell;
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    cell = &buffer_[pos & buffer_mask_];
    size_t seq = cell->sequence_.load(std::memory_order_acquire);
    intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
    if (dif == 0) {
      if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        data = std::move(cell->data_);
        cell->sequence_.store(pos + buffer_mask_ + 1, std::memory_order_release);
        return true;
      }
    }
    return false;
  }

  inline size_t size() const {
    return enqueue_pos_ - dequeue_pos_;
  }

  inline bool empty() {
    return size() == 0;
  }

 protected:
  struct cell_t {
    std::atomic<size_t> sequence_;
    T data_;
  };

  static size_t const cacheline_size = 64;
  typedef char cacheline_pad_t[cacheline_size];

  cacheline_pad_t pad0_;
  cell_t* const buffer_;
  size_t const buffer_mask_;
  cacheline_pad_t pad1_;
  std::atomic<size_t> enqueue_pos_;
  cacheline_pad_t pad2_;
  std::atomic<size_t> dequeue_pos_;
  cacheline_pad_t pad3_;

  MPMCBoundedQueue(MPMCBoundedQueue const&) = delete;
  void operator=(MPMCBoundedQueue const&) = delete;
};

}  // namespace runtime
}  // namespace matxscript
