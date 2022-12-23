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

#include <matxscript/runtime/threadpool/mpmc_bounded_queue.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace byted_matx_vision {
namespace ops {
namespace vision {

template <class T>
class BoundedObjectPool : public std::enable_shared_from_this<BoundedObjectPool<T>> {
  using TSelf = BoundedObjectPool<T>;
  using LockFreeQueue = ::matxscript::runtime::MPMCBoundedQueue<T*>;

 public:
  using std::enable_shared_from_this<TSelf>::shared_from_this;
  BoundedObjectPool(std::vector<std::unique_ptr<T>> objects)
      : queue_(GetCapacitySize(objects.size())) {
    objects_ = std::move(objects);
    for (auto& obj : objects_) {
      queue_.enqueue(obj.get());
    }
  }
  ~BoundedObjectPool() = default;

  std::shared_ptr<T> borrow() {
    T* obj_ptr = nullptr;
    for (;;) {
      if (queue_.try_dequeue(obj_ptr)) {
        break;
      }
      // queue is empty, sleep 1us
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
    return std::shared_ptr<T>(obj_ptr, ObjectDeleter(shared_from_this()));
  }

 protected:
  class ObjectDeleter {
   public:
    explicit ObjectDeleter(std::shared_ptr<TSelf> pool) : pool_(std::move(pool)) {
    }
    void operator()(T* obj_ptr) {
      pool_->queue_.enqueue(obj_ptr);
    }

   private:
    std::shared_ptr<TSelf> pool_;
  };

  static size_t GetCapacitySize(size_t cap) {
    size_t length = 1;
    while (length < cap) {
      length <<= 1;
    }
    return length;
  }

 protected:
  LockFreeQueue queue_;
  std::vector<std::unique_ptr<T>> objects_;
};

}  // namespace vision
}  // namespace ops
}  // namespace byted_matx_vision
