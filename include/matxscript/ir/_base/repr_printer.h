// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
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

#include <iostream>

#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/*! \brief A printer class to print the AST/IR nodes. */
class ReprPrinter {
 public:
  /*! \brief The output stream */
  std::ostream& stream;
  /*! \brief The indentation level. */
  int indent{0};

  explicit ReprPrinter(std::ostream& stream)  // NOLINT(*)
      : stream(stream) {
  }

  /*! \brief The value to be printed. */
  MATX_DLL void Print(const RTValue& value);
  /*! \brief The node to be printed. */
  MATX_DLL void Print(const ObjectRef& node);
  /*! \brief Print indent to the stream */
  MATX_DLL void PrintIndent();
  // Allow registration to be printer.
  using FType = NodeFunctor<void(const ObjectRef&, ReprPrinter*)>;
  MATX_DLL static FType& vtable();
};

/*!
 * \brief Dump the node to stderr, used for debug purposes.
 * \param node The input node
 */
MATX_DLL void Dump(const ObjectRef& node);

/*!
 * \brief Dump the node to stderr, used for debug purposes.
 * \param node The input node
 */
MATX_DLL void Dump(const Object* node);

// default print function for all objects
// provide in the runtime namespace as this is where objectref originally comes from.
inline std::ostream& operator<<(std::ostream& os, const ObjectRef& n) {  // NOLINT(*)
  ReprPrinter(os).Print(n);
  return os;
}

}  // namespace runtime
}  // namespace matxscript
