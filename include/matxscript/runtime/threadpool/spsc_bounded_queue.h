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
class SPSCBoundedQueue {
 public:
  SPSCBoundedQueue(size_t size)
      : _size(size),
        _mask(size - 1),
        _buffer(
            reinterpret_cast<T*>(new aligned_t[_size + 1])),  // need one extra element for a guard
        _head(0),
        _tail(0) {
    // make sure it's a power of 2
    if (!((_size != 0) && ((_size & (~_size + 1)) == _size))) {
      abort();
    }
  }

  ~SPSCBoundedQueue() {
    delete[] _buffer;
  }

  bool enqueue(T& input) {
    const size_t head = _head.load(std::memory_order_relaxed);

    if (((_tail.load(std::memory_order_acquire) - (head + 1)) & _mask) >= 1) {
      _buffer[head & _mask] = input;
      _head.store(head + 1, std::memory_order_release);
      return true;
    }
    return false;
  }

  bool dequeue(T& output) {
    const size_t tail = _tail.load(std::memory_order_relaxed);

    if (((_head.load(std::memory_order_acquire) - tail) & _mask) >= 1) {
      output = _buffer[_tail & _mask];
      _tail.store(tail + 1, std::memory_order_release);
      return true;
    }
    return false;
  }

 private:
  typedef typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type aligned_t;
  typedef char cache_line_pad_t[64];

  cache_line_pad_t _pad0;
  const size_t _size;
  const size_t _mask;
  T* const _buffer;

  cache_line_pad_t _pad1;
  std::atomic<size_t> _head;

  cache_line_pad_t _pad2;
  std::atomic<size_t> _tail;

  SPSCBoundedQueue(const SPSCBoundedQueue&) = delete;
  void operator=(const SPSCBoundedQueue&) = delete;
};

}  // namespace runtime
}  // namespace matxscript
