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
#include <memory>

namespace matxscript {
namespace runtime {

class Symbol;

class InterpreterOp : public OpKernel {
  enum class OpCode {
    // arith ops
    OP_CODE_BEGIN = 0,
    __add__ = 0,
    __sub__,
    __mul__,
    __floordiv__,
    __truediv__,
    __mod__,
    __neg__,
    __pos__,
    __pow__,
    __abs__,
    __index__,

    // logical ops
    __gt__,
    __ge__,
    __lt__,
    __le__,
    __eq__,
    __ne__,

    // bitwise ops
    __invert__,
    __lshift__,
    __rshift__,
    __xor__,
    __and__,
    __or__,

    // functions
    __bool__,
    __len__,
    __contains__,
    __setitem__,
    __getitem__,
    __delitem__,
    __setslice__,
    __getslice__,
    __delslice__,
    __iter__,
    __next__,
    __getattr__,
    __setattr__,
    __call__,

    // container
    ListConstructor,
    DictConstructor,
    SetConstructor,
    TupleConstructor,

    // AutoParallel
    ParallelMap,
    ParallelStarMap,
    ApplyAsync,

    __iter_and_check_len__,

    OP_CODE_END,
  };

  static string_view OpCode2Str(int op_code);
  static int Str2OpCode(const string_view& op_name);

 public:
  InterpreterOp() = default;
  ~InterpreterOp() override = default;

  void Init() override;

  String GetHumanName(bool with_debug_info) const;

 public:
  RTValue Process(PyArgs inputs) const override;

 protected:
  String GenDebugMessage() const;

 private:
  int opcode_ = -1;
  int64_t py_source_line_ = -1;
  String py_source_file_;
  String py_source_func_;
  String py_source_stmt_;
};

}  // namespace runtime
}  // namespace matxscript
