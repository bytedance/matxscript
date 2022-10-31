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

#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Args like Python *args
 *
 * PyArgs does not hold the life cycle of the object, users need to maintain it by themselves.
 *
 * PyArgs cannot be constructed independently. Only the following way are supported
 *
 * void my_func(PyArgs args);
 *
 * 1. full-expression
 *    my_func({arg_0, arg_1, ..., arg_n})
 *
 *    The C++ Standard guarantees it to work. The Standard mandates that all temporary objects get
 *    destroyed as the last step of evaluating of the full-expression that contains the point where
 *    the temporaries were created1. "full expression" means an expression that is not
 *    sub-expression of other expressions.
 *
 * 2. std::vector
 *    std::vector<> args_init;
 *    args_init.push_back(...)
 *    ...
 *    my_func(PyArgs(args_init.data(), args_init.size()));
 *
 *****************************************************************************/

struct PyArgs {
 public:
  constexpr PyArgs() : item_addr_(nullptr), size_(0) {
  }
  constexpr PyArgs(std::initializer_list<RTView> args)
      : item_addr_(const_cast<Any*>(static_cast<const Any*>(args.begin()))), size_(args.size()) {
  }
  constexpr explicit PyArgs(const Any* begin, size_t len)
      : item_addr_(const_cast<Any*>(begin)), size_(len) {
  }
  PyArgs(const PyArgs& other) = default;
  PyArgs(PyArgs&& other) = default;
  PyArgs& operator=(const PyArgs& other) = default;
  PyArgs& operator=(PyArgs&& other) = default;
  constexpr int size() const {
    return size_;
  }
  constexpr const Any* begin() const {
    return item_addr_;
  }
  constexpr const Any* end() const {
    return item_addr_ + size_;
  }
  // user should check i is valid
  inline const Any& operator[](int64_t i) const {
    return *(item_addr_ + i);
  }

  // https://en.cppreference.com/w/cpp/utility/initializer_list
  // user can move args[i] for each element in initializer_list is copy-initialized
  inline Any& operator[](int64_t i) {
    return *(item_addr_ + i);
  }

 private:
  Any* item_addr_;
  size_t size_;
};

}  // namespace runtime
}  // namespace matxscript
