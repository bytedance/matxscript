// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 *
 * Copyright (c) 2015 by Contributors
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

#include "runtime_port.h"

#include <memory>
#include <mutex>
#include <vector>

namespace matxscript {
namespace runtime {

// macro hanlding for threadlocal variables
#ifdef __GNUC__
#define MX_THREAD_LOCAL __thread
#elif __STDC_VERSION__ >= 201112L
#define MX_THREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
#define MX_THREAD_LOCAL __declspec(thread)
#endif

#if MATXSCRIPT_CXX11_THREAD_LOCAL == 0
#pragma message("Warning: CXX11 thread_local is not formally supported")
#endif

/*!
 * \brief A threadlocal store to store threadlocal variables.
 *  Will return a thread local singleton of type T
 * \tparam T the type we like to store
 */
template <typename T>
class ThreadLocalStore {
 public:
  /*! \return get a thread local singleton */
  static T* Get() {
#if MATXSCRIPT_CXX11_THREAD_LOCAL && MATXSCRIPT_MODERN_THREAD_LOCAL == 1
    static thread_local T inst;
    return &inst;
#else
    static MX_THREAD_LOCAL T* ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      // Syntactic work-around for the nvcc of the initial cuda v10.1 release,
      // which fails to compile 'Singleton()->' below. Fixed in v10.1 update 1.
      (*Singleton()).RegisterDelete(ptr);
    }
    return ptr;
#endif
  }

 private:
  /*! \brief constructor */
  ThreadLocalStore() {
  }
  /*! \brief destructor */
  ~ThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  /*! \return singleton of the store */
  static ThreadLocalStore<T>* Singleton() {
    static ThreadLocalStore<T> inst;
    return &inst;
  }
  /*!
   * \brief register str for internal deletion
   * \param str the string pointer
   */
  void RegisterDelete(T* str) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(str);
    lock.unlock();
  }
  /*! \brief internal mutex */
  std::mutex mutex_;
  /*!\brief internal data */
  std::vector<T*> data_;
};

}  // namespace runtime
}  // namespace matxscript
