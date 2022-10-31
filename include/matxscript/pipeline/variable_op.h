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

#include <matxscript/pipeline/op_kernel.h>

namespace matxscript {
namespace runtime {

// const static char* VARIABLE_OP_NAME = "VariableOp";

class Symbol;

class VariableOp : public OpKernel {
 public:
  VariableOp() : OpKernel() {
    class_name_ = "VariableOp";
    name_ = "VariableOp_0";
  }
  static std::unique_ptr<Symbol> make_symbol(OpKernelPtr op, String name, runtime::RTValue data);

  RTValue Process(PyArgs inputs) const override {
    CheckArgs(inputs.size(), 1);
    return inputs[0].As<RTValue>();
  }
};

}  // namespace runtime
}  // namespace matxscript
