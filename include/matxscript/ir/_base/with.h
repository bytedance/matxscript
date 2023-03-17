// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from TVM.
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

/*!
 * \file matxscript/ir/_base/with.h
 * \brief RAII wrapper function to enter and exit a context object
 *        similar to python's with syntax.
 */
#pragma once

#include <functional>
#include <utility>

namespace matxscript {
namespace ir {

/*!
 * \brief RAII wrapper function to enter and exit a context object
 *        similar to python's with syntax.
 *
 * \code
 * // context class
 * class MyContext {
 *  private:
 *    friend class With<MyContext>;
      MyContext(arguments);
 *    void EnterWithScope();
 *    void ExitWithScope();
 * };
 *
 * {
 *   With<MyContext> scope(arguments);
 *   // effect take place.
 * }
 * \endcode
 *
 * \tparam ContextType Type of the context object.
 */
template <typename ContextType>
class With {
 public:
  /*!
   * \brief constructor.
   *  Enter the scope of the context.
   */
  template <typename... Args>
  explicit With(Args&&... args) : ctx_(std::forward<Args>(args)...) {
    ctx_.EnterWithScope();
  }
  /*! \brief destructor, leaves the scope of the context. */
  ~With() noexcept(false) {
    ctx_.ExitWithScope();
  }

  // Disable copy and move construction.  `With` is intended only for
  // use in nested contexts that are exited in the reverse order of
  // entry.  Allowing context to be copied or moved would break this
  // expectation.
  With(const With& other) = delete;
  With& operator=(const With& other) = delete;
  With(With&& other) = delete;
  With& operator=(With&& other) = delete;

  ContextType* get() {
    return &ctx_;
  }
  const ContextType* get() const {
    return &ctx_;
  }

  ContextType* operator->() {
    return get();
  }
  const ContextType* operator->() const {
    return get();
  }
  ContextType& operator*() {
    return *get();
  }
  const ContextType* operator*() const {
    return *get();
  }

  ContextType operator()() {
    return ctx_;
  }

 private:
  /*! \brief internal context type. */
  ContextType ctx_;
};

}  // namespace ir
}  // namespace matxscript
