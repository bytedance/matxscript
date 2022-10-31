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
#include <matxscript/pipeline/interpreter_op.h>
#include <matxscript/pipeline/node.h>
#include <matxscript/pipeline/symbolic_executor.h>
#include <matxscript/runtime/container/dict_private.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/generic/generic_funcs.h>
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(InterpreterOp);

string_view InterpreterOp::OpCode2Str(int op_code) {
  if (op_code < int(OpCode::OP_CODE_BEGIN) || op_code >= int(OpCode::OP_CODE_END)) {
    return "unknown";
  }
  switch (static_cast<OpCode>(op_code)) {
    case OpCode::__add__: {
      return "__add__";
    } break;
    case OpCode::__sub__: {
      return "__sub__";
    } break;
    case OpCode::__mul__: {
      return "__mul__";
    } break;
    case OpCode::__floordiv__: {
      return "__floordiv__";
    } break;
    case OpCode::__truediv__: {
      return "__truediv__";
    } break;
    case OpCode::__mod__: {
      return "__mod__";
    } break;
    case OpCode::__neg__: {
      return "__neg__";
    } break;
    case OpCode::__pos__: {
      return "__pos__";
    } break;
    case OpCode::__pow__: {
      return "__pow__";
    } break;
    case OpCode::__abs__: {
      return "__abs__";
    } break;
    case OpCode::__index__: {
      return "__index__";
    } break;

    // logical ops
    case OpCode::__gt__: {
      return "__gt__";
    } break;
    case OpCode::__ge__: {
      return "__ge__";
    } break;
    case OpCode::__lt__: {
      return "__lt__";
    } break;
    case OpCode::__le__: {
      return "__le__";
    } break;
    case OpCode::__eq__: {
      return "__eq__";
    } break;
    case OpCode::__ne__: {
      return "__ne__";
    } break;

    // bitwise ops
    case OpCode::__invert__: {
      return "__invert__";
    } break;
    case OpCode::__lshift__: {
      return "__lshift__";
    } break;
    case OpCode::__rshift__: {
      return "__rshift__";
    } break;
    case OpCode::__xor__: {
      return "__xor__";
    } break;
    case OpCode::__and__: {
      return "__and__";
    } break;
    case OpCode::__or__: {
      return "__or__";
    } break;

    // functions
    case OpCode::__bool__: {
      return "__bool__";
    } break;
    case OpCode::__len__: {
      return "__len__";
    } break;
    case OpCode::__contains__: {
      return "__contains__";
    } break;
    case OpCode::__setitem__: {
      return "__setitem__";
    } break;
    case OpCode::__getitem__: {
      return "__getitem__";
    } break;
    case OpCode::__delitem__: {
      return "__delitem__";
    } break;
    case OpCode::__setslice__: {
      return "__setslice__";
    } break;
    case OpCode::__getslice__: {
      return "__getslice__";
    } break;
    case OpCode::__delslice__: {
      return "__delslice__";
    } break;
    case OpCode::__getattr__: {
      return "__getattr__";
    } break;
    case OpCode::__setattr__: {
      return "__setattr__";
    } break;
    case OpCode::__call__: {
      return "__call__";
    } break;
    case OpCode::__iter__: {
      return "__iter__";
    } break;
    case OpCode::__iter_and_check_len__: {
      return "__iter_and_check_len__";
    } break;
    case OpCode::__next__: {
      return "__next__";
    } break;
    case OpCode::ListConstructor: {
      return "ListConstructor";
    } break;
    case OpCode::DictConstructor: {
      return "DictConstructor";
    } break;
    case OpCode::SetConstructor: {
      return "SetConstructor";
    } break;
    case OpCode::TupleConstructor: {
      return "TupleConstructor";
    } break;
    case OpCode::ParallelMap: {
      return "ParallelMap";
    } break;
    case OpCode::ParallelStarMap: {
      return "ParallelStarMap";
    } break;
    case OpCode::ApplyAsync: {
      return "ApplyAsync";
    } break;
    default: {
      return "unknown";
    } break;
  }
}

int InterpreterOp::Str2OpCode(const string_view& op_name) {
  if (op_name == "__add__") {
    return int(OpCode::__add__);
  } else if (op_name == "__sub__") {
    return int(OpCode::__sub__);
  } else if (op_name == "__mul__") {
    return int(OpCode::__mul__);
  } else if (op_name == "__floordiv__") {
    return int(OpCode::__floordiv__);
  } else if (op_name == "__truediv__") {
    return int(OpCode::__truediv__);
  } else if (op_name == "__mod__") {
    return int(OpCode::__mod__);
  } else if (op_name == "__neg__") {
    return int(OpCode::__neg__);
  } else if (op_name == "__pos__") {
    return int(OpCode::__pos__);
  } else if (op_name == "__pow__") {
    return int(OpCode::__pow__);
  } else if (op_name == "__abs__") {
    return int(OpCode::__abs__);
  } else if (op_name == "__index__") {
    return int(OpCode::__index__);
  }

  // logical ops
  else if (op_name == "__gt__") {
    return int(OpCode::__gt__);
  } else if (op_name == "__ge__") {
    return int(OpCode::__ge__);
  } else if (op_name == "__lt__") {
    return int(OpCode::__lt__);
  } else if (op_name == "__le__") {
    return int(OpCode::__le__);
  } else if (op_name == "__eq__") {
    return int(OpCode::__eq__);
  } else if (op_name == "__ne__") {
    return int(OpCode::__ne__);
  }

  // bitwise ops
  else if (op_name == "__invert__") {
    return int(OpCode::__invert__);
  } else if (op_name == "__lshift__") {
    return int(OpCode::__lshift__);
  } else if (op_name == "__rshift__") {
    return int(OpCode::__rshift__);
  } else if (op_name == "__xor__") {
    return int(OpCode::__xor__);
  } else if (op_name == "__and__") {
    return int(OpCode::__and__);
  } else if (op_name == "__or__") {
    return int(OpCode::__or__);
  }

  // functions
  else if (op_name == "__bool__") {
    return int(OpCode::__bool__);
  } else if (op_name == "__len__") {
    return int(OpCode::__len__);
  } else if (op_name == "__contains__") {
    return int(OpCode::__contains__);
  } else if (op_name == "__setitem__") {
    return int(OpCode::__setitem__);
  } else if (op_name == "__getitem__") {
    return int(OpCode::__getitem__);
  } else if (op_name == "__delitem__") {
    return int(OpCode::__delitem__);
  } else if (op_name == "__setslice__") {
    return int(OpCode::__setslice__);
  } else if (op_name == "__getslice__") {
    return int(OpCode::__getslice__);
  } else if (op_name == "__delslice__") {
    return int(OpCode::__delslice__);
  } else if (op_name == "__getattr__") {
    return int(OpCode::__getattr__);
  } else if (op_name == "__setattr__") {
    return int(OpCode::__setattr__);
  } else if (op_name == "__call__") {
    return int(OpCode::__call__);
  } else if (op_name == "__iter__") {
    return int(OpCode::__iter__);
  } else if (op_name == "__iter_and_check_len__") {
    return int(OpCode::__iter_and_check_len__);
  } else if (op_name == "__next__") {
    return int(OpCode::__next__);
  }

  // container
  else if (op_name == "ListConstructor") {
    return int(OpCode::ListConstructor);
  } else if (op_name == "DictConstructor") {
    return int(OpCode::DictConstructor);
  } else if (op_name == "SetConstructor") {
    return int(OpCode::SetConstructor);
  } else if (op_name == "TupleConstructor") {
    return int(OpCode::TupleConstructor);
  } else if (op_name == "ParallelMap") {
    return int(OpCode::ParallelMap);
  } else if (op_name == "ParallelStarMap") {
    return int(OpCode::ParallelStarMap);
  } else if (op_name == "ApplyAsync") {
    return int(OpCode::ApplyAsync);
  } else {
    return -1;
  }
}

void InterpreterOp::Init() {
  auto opcode = GetAttr<Unicode>("opcode").encode();
  opcode_ = Str2OpCode(opcode);
  // optional python source debug info
  if (HasAttr("py_source_file")) {
    py_source_file_ = GetAttr<Unicode>("py_source_file").encode();
  }
  if (HasAttr("py_source_line")) {
    py_source_line_ = GetAttr<int64_t>("py_source_line");
  }
  if (HasAttr("py_source_func")) {
    py_source_func_ = GetAttr<Unicode>("py_source_func").encode();
  }
  if (HasAttr("py_source_stmt")) {
    py_source_stmt_ = GetAttr<Unicode>("py_source_stmt").encode();
  }
}

String InterpreterOp::GenDebugMessage() const {
  if (py_source_line_ < 0) {
    return {};
  }
  String message;
  auto py_source_line = std::to_string(py_source_line_);
  message.reserve(1024);
  message.append("File \"").append(py_source_file_).append("\", ");
  message.append("line ").append(py_source_line.data(), py_source_line.size());
  message.append(", in ").append(py_source_func_).append("\n");
  message.append("  ").append(py_source_stmt_);
  return message;
}

RTValue InterpreterOp::Process(PyArgs inputs) const {
  if (opcode_ < int(OpCode::OP_CODE_BEGIN) || opcode_ >= int(OpCode::OP_CODE_END)) {
    MXTHROW << "[InterpreterOp::Process] unknown op_code: " << opcode_;
  }
  auto code = OpCode(opcode_);
  switch (code) {
    case OpCode::__add__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::add(inputs[0], inputs[1]);
    } break;
    case OpCode::__sub__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::sub(inputs[0], inputs[1]);
    } break;
    case OpCode::__mul__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::mul(inputs[0], inputs[1]);
    } break;
    case OpCode::__floordiv__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::floordiv(inputs[0], inputs[1]);
    } break;
    case OpCode::__truediv__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return RTValue(inputs[0].As<double>() / inputs[1].As<double>());
    } break;
    case OpCode::__mod__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__neg__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__pos__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__pow__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__abs__: {
      MXCHECK(inputs.size() == 1) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 1 arguments but get " << inputs.size();
      return ArithOps::abs(inputs[0]);
    } break;
    case OpCode::__index__: {
      MXCHECK(inputs.size() == 1) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 1 arguments but get " << inputs.size();
      return Kernel_int64_t::make(inputs[0]);
    } break;

    // logical ops
    case OpCode::__gt__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::gt(inputs[0], inputs[1]);
    } break;
    case OpCode::__ge__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::ge(inputs[0], inputs[1]);
    } break;
    case OpCode::__lt__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::lt(inputs[0], inputs[1]);
    } break;
    case OpCode::__le__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::le(inputs[0], inputs[1]);
    } break;
    case OpCode::__eq__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::eq(inputs[0], inputs[1]);
    } break;
    case OpCode::__ne__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return ArithOps::ne(inputs[0], inputs[1]);
    } break;

    // bitwise ops
    case OpCode::__invert__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__lshift__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__rshift__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__xor__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__and__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__or__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;

    // functions
    case OpCode::__bool__: {
      MXCHECK(inputs.size() == 1) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 1 arguments but get " << inputs.size();
      return Kernel_bool::make(inputs[0]);
    } break;
    case OpCode::__len__: {
      MXCHECK(inputs.size() == 1) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 1 arguments but get "
                                  << inputs.size();
      try {
        return kernel_object___len__(inputs[0]);
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: object of type '",
                           inputs[0].type_name(),
                           "' has no len()");
      }
    } break;
    case OpCode::__contains__: {
      MXCHECK(inputs.size() == 2) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 2 arguments but get " << inputs.size();
      return kernel_object___contains__(inputs[0], inputs[1]);
    } break;
    case OpCode::__setitem__: {
      MXCHECK(inputs.size() == 3) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 3 arguments but get " << inputs.size();
      return kernel_object___setitem__(inputs[0], inputs[1], inputs[2]);
    } break;
    case OpCode::__getitem__: {
      MXCHECK(inputs.size() == 2) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 2 arguments but get "
                                  << inputs.size();
      try {
        return kernel_object___getitem__(inputs[0], inputs[1]);
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute __getitem__ failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute __getitem__ failed: internal error");
      }
    } break;
    case OpCode::__delitem__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__setslice__: {
      MXCHECK(inputs.size() == 4) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 4 arguments but get " << inputs.size();
      return kernel_object___setslice__(inputs[0], inputs[1], inputs[2], inputs[3]);
    } break;
    case OpCode::__getslice__: {
      try {
        if (inputs.size() == 3) {
          return kernel_object___getslice__(inputs[0], inputs[1], inputs[2], RTView(1));
        } else if (inputs.size() == 4) {
          return kernel_object___getslice__(inputs[0], inputs[1], inputs[2], inputs[3]);
        }
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute __getitem__ failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute __getitem__ failed: internal error");
      }
      MXTHROW << this->GenDebugMessage()
              << "\nTypeError: [InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
              << "] Expect 3 or 4 arguments but get " << inputs.size();
    } break;
    case OpCode::__delslice__: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
    case OpCode::__getattr__: {
      MXCHECK(inputs.size() == 2) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 2 arguments but get "
                                  << inputs.size();
      if (!inputs[0].IsObjectRef<UserDataRef>()) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: ",
                           inputs[0].type_name(),
                           ".",
                           inputs[1].As<string_view>(),
                           " is not supported by matx.trace");
      }
      try {
        return kernel_object___getattr__(inputs[0], inputs[1].As<string_view>());
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute __getattr__ failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute __getattr__ failed: internal error");
      }
    } break;
    case OpCode::__setattr__: {
      MXCHECK(inputs.size() == 3) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 3 arguments but get "
                                  << inputs.size();
      try {
        return kernel_object___setattr__(inputs[0], inputs[1].As<string_view>(), inputs[2]);
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute __setattr__ failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute __setattr__ failed: internal error");
      }
    } break;
    case OpCode::__call__: {
      MXCHECK(inputs.size() >= 1) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 1 or more arguments but get "
                                  << inputs.size();
      if (!inputs[0].IsObjectRef<UserDataRef>()) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: '",
                           inputs[0].type_name(),
                           "' object is not callable");
      }
      try {
        auto ud_view = inputs[0].AsObjectViewNoCheck<UserDataRef>();
        return ud_view.data().generic_call(PyArgs(inputs.begin() + 1, inputs.size() - 1));
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute __call__ failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute __call__ failed: internal error");
      }
    } break;
    case OpCode::__iter__: {
      MXCHECK(inputs.size() == 1) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 1 arguments but get " << inputs.size();
      return Kernel_Iterable::make(inputs[0]);
    } break;
    case OpCode::__iter_and_check_len__: {
      MXCHECK(inputs.size() == 2) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 2 arguments but get "
                                  << inputs.size();
      auto expect_len = inputs[1].As<int64_t>();
      auto real_len = kernel_object___len__(inputs[0]);
      if (expect_len < real_len) {
        THROW_PY_ValueError(this->GenDebugMessage(),
                            "\nValueError: too many values to unpack (expected ",
                            expect_len,
                            ", got ",
                            real_len,
                            "). The expected num is a snapshot during tracing ",
                            "in order to ensure data consistency");
      } else if (expect_len > real_len) {
        THROW_PY_ValueError(this->GenDebugMessage(),
                            "\nValueError: not enough values to unpack (expected ",
                            expect_len,
                            ", got ",
                            real_len,
                            "). The expected num is a snapshot during tracing ",
                            "in order to ensure data consistency");
      }
      try {
        return Kernel_Iterable::make(inputs[0]);
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute __iter__ failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute __iter__ failed: internal error");
      }
    } break;
    case OpCode::__next__: {
      // the second argument is the last element for fix trace order
      MXCHECK(inputs.size() == 1 || inputs.size() == 2)
          << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
          << "] Expect 1 or 2 arguments but get " << inputs.size();
      auto iterable_view = inputs[0].AsObjectView<Iterator>();
      if (!iterable_view.data().HasNext()) {
        THROW_PY_StopIteration(this->GenDebugMessage());
      }
      return iterable_view.data().Next();
    } break;
    case OpCode::ListConstructor: {
      return List(inputs.begin(), inputs.end());
    } break;
    case OpCode::DictConstructor: {
      MXCHECK(inputs.size() % 2 == 0)
          << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
          << "] Expect even number of arguments but get " << inputs.size();
      Dict ret;
      ret.reserve(inputs.size() / 2);
      for (size_t i = 0; i < inputs.size(); i += 2) {
        ret.emplace(inputs[i].As<RTValue>(), inputs[i + 1].As<RTValue>());
      }
      return ret;
    } break;
    case OpCode::SetConstructor: {
      return Set(inputs.begin(), inputs.end());
    } break;
    case OpCode::TupleConstructor: {
      return Tuple(inputs.begin(), inputs.end());
    } break;
    case OpCode::ParallelMap: {
      MXCHECK(inputs.size() == 2) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 2 arguments but get "
                                  << inputs.size();
      try {
        auto user_func = inputs[0].AsObjectView<UserDataRef>();
        return ParallelMap(user_func.data(), inputs[1], belong_to_);
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute matx.pmap failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute matx.pmap failed: internal error");
      }
    } break;
    case OpCode::ParallelStarMap: {
      MXCHECK(inputs.size() == 2) << this->GenDebugMessage()
                                  << "\nTypeError: [InterpreterOp::Process][opcode: "
                                  << OpCode2Str(opcode_) << "] Expect 2 arguments but get "
                                  << inputs.size();
      try {
        auto user_func = inputs[0].AsObjectView<UserDataRef>();
        return ParallelStarMap(user_func.data(), inputs[1], belong_to_);
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute matx.pstarmap failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute matx.pstarmap failed: internal error");
      }
    } break;
    case OpCode::ApplyAsync: {
      MXCHECK(inputs.size() >= 1) << "[InterpreterOp::Process][opcode: " << OpCode2Str(opcode_)
                                  << "] Expect 1 or more arguments but get " << inputs.size()
                                  << "\n"
                                  << this->GenDebugMessage();
      try {
        auto user_func = inputs[0].AsObjectView<UserDataRef>();
        return ApplyAsync(
            user_func.data(), PyArgs(inputs.begin() + 1, inputs.size() - 1), belong_to_);
      } catch (const std::exception& e) {
        THROW_PY_TypeError(
            this->GenDebugMessage(), "\nTypeError: execute matx.apply_async failed: ", e.what());
      } catch (...) {
        THROW_PY_TypeError(this->GenDebugMessage(),
                           "\nTypeError: execute matx.apply_async failed: internal error");
      }
    } break;
    default: {
      MXTHROW << "[InterpreterOp::Process] unsupported opcode: " << opcode_
              << ", name: " << OpCode2Str(opcode_);
    } break;
  }
  return None;
}

}  // namespace runtime
}  // namespace matxscript
