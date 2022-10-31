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

#include <stdlib.h>

#include <mutex>

namespace matxscript {
namespace runtime {

template <class T>
class Singleton {
 public:
  friend T;

  static T* instance() {
    if (_s_instance == NULL) {
      std::lock_guard<std::mutex> lock(s_singleton_mutex_);
      if (_s_instance == NULL) {
        _s_instance = new T;
      }
    }
    return _s_instance;
  }

 private:
  static T* _s_instance;
  static std::mutex s_singleton_mutex_;
};

template <class T>
T* Singleton<T>::_s_instance = NULL;

template <class T>
std::mutex Singleton<T>::s_singleton_mutex_;

}  // namespace runtime
}  // namespace matxscript
