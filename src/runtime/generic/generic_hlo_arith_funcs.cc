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
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/unicode_view.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/global_type_index.h>

namespace matxscript {
namespace runtime {

RTValue ArithOps::add(const Any& lhs, const Any& rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::add(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::add(lhs.value_.data.v_float64, rhs);
    } break;
    case TypeIndex::kRuntimeString: {
      return ArithOps::add(lhs.AsNoCheck<string_view>(), rhs);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return ArithOps::add(lhs.AsNoCheck<unicode_view>(), rhs);
    } break;
    case TypeIndex::kRuntimeList: {
      return ArithOps::add(lhs.AsNoCheck<List>(), rhs);
    } break;
    case TypeIndex::kRuntimeFTList: {
      return lhs.AsObjectViewNoCheck<FTObjectBase>().data().generic_call_attr("__add__",
                                                                              {rhs.As<RTView>()});
    } break;
    case TypeIndex::kRuntimeTuple: {
      return ArithOps::add(lhs.AsObjectViewNoCheck<Tuple>().data(), rhs);
    } break;
    default: {
      THROW_PY_TypeError(
          "unsupported operand type(s) for +: '", lhs.type_name(), "' and '", rhs.type_name(), "'");
    } break;
  }
  return None;
}

RTValue ArithOps::mul(const Any& lhs, const Any& rhs) {
  if (rhs.type_code() == TypeIndex::kRuntimeFTList) {
    return rhs.AsObjectViewNoCheck<FTObjectBase>().data().generic_call_attr("__mul__",
                                                                            PyArgs(&lhs, 1));
  }
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::mul(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::mul(lhs.value_.data.v_float64, rhs);
    } break;
    case TypeIndex::kRuntimeString: {
      return ArithOps::mul(lhs.AsNoCheck<string_view>(), rhs);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return ArithOps::mul(lhs.AsNoCheck<unicode_view>(), rhs);
    } break;
    case TypeIndex::kRuntimeList: {
      return ArithOps::mul(lhs.AsObjectViewNoCheck<List>().data(), rhs);
    } break;
    case TypeIndex::kRuntimeFTList: {
      return lhs.AsObjectViewNoCheck<FTObjectBase>().data().generic_call_attr("__mul__",
                                                                              PyArgs(&rhs, 1));
    } break;
    case TypeIndex::kRuntimeTuple: {
      return ArithOps::mul(lhs.AsObjectViewNoCheck<Tuple>().data(), rhs);
    } break;
    default: {
      THROW_PY_TypeError(
          "unsupported operand type(s) for *: '", lhs.type_name(), "' and '", rhs.type_name(), "'");
    } break;
  }
  return 0;
}

RTValue ArithOps::sub(int64_t lhs, const Any& rhs) {
  switch (rhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return RTValue(lhs - rhs.value_.data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return RTValue(lhs - rhs.value_.data.v_float64);
    } break;
    default: {
      THROW_PY_TypeError("unsupported operand type(s) for -: 'int' and '", rhs.type_name(), "'");
    } break;
  }
  return 0;
}

RTValue ArithOps::sub(const Any& lhs, int64_t rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return lhs.value_.data.v_int64 - rhs;
    } break;
    case TypeIndex::kRuntimeFloat: {
      return lhs.value_.data.v_float64 - rhs;
    } break;
    default: {
      THROW_PY_TypeError("unsupported operand type(s) for -: '", lhs.type_name(), "' and 'int'");
    } break;
  }
  return 0;
}

RTValue ArithOps::sub(const Any& lhs, const Any& rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::sub(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::sub(lhs.value_.data.v_float64, rhs);
    } break;
    default: {
      THROW_PY_TypeError(
          "unsupported operand type(s) for -: '", lhs.type_name(), "' and '", rhs.type_name(), "'");
    } break;
  }
  return 0;
}

RTValue ArithOps::abs(const Any& x) {
  switch (x.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return std::abs(x.value_.data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return fabs(x.value_.data.v_float64);
    } break;
    default: {
      THROW_PY_TypeError("bad operand type for abs(): '", x.type_name(), "'");
    } break;
  }
  return 0;
}

RTValue ArithOps::floordiv(int64_t lhs, const Any& rhs) {
  switch (rhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::floordiv(lhs, rhs.value_.data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      double result = std::floor(static_cast<double>(lhs) / rhs.value_.data.v_float64);
      MXCHECK(!std::isnan(result) && !std::isinf(result)) << "ValueError: math domain error";
      return result;
    } break;
    default: {
      THROW_PY_TypeError("unsupported operand type(s) for //: 'int' and '", rhs.type_name(), "'");
    } break;
  }
  return 0;
}

RTValue ArithOps::floordiv(const Any& lhs, int64_t rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::floordiv(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      double result = std::floor(lhs.value_.data.v_float64 / static_cast<double>(rhs));
      if (std::isnan(result) || std::isinf(result)) {
        THROW_PY_ValueError("math domain error");
      }
      return result;
    } break;
    default: {
      THROW_PY_TypeError("unsupported operand type(s) for //: '", lhs.type_name(), "' and 'int'");
    } break;
  }
  return 0;
}

RTValue ArithOps::floordiv(const Any& lhs, const Any& rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::floordiv(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::floordiv(lhs.value_.data.v_float64, rhs);
    } break;
    default: {
      THROW_PY_TypeError("unsupported operand type(s) for //: '",
                         lhs.type_name(),
                         "' and '",
                         rhs.type_name(),
                         "'");
    } break;
  }
  return 0;
}

RTValue ArithOps::floormod(int64_t lhs, const Any& rhs) {
  switch (rhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::floormod(lhs, rhs.value_.data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      double result = fmod(static_cast<double>(lhs), rhs.value_.data.v_float64);
      if (std::isnan(result) || std::isinf(result)) {
        THROW_PY_ValueError("math domain error");
      }
      return result;
    } break;
    default: {
      THROW_PY_TypeError("unsupported operand type(s) for %: 'int' and '", rhs.type_name(), "'");
    } break;
  }
  return 0;
}

RTValue ArithOps::floormod(const Any& lhs, int64_t rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::floormod(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      double result = fmod(lhs.value_.data.v_float64, static_cast<double>(rhs));
      if (std::isnan(result) || std::isinf(result)) {
        THROW_PY_ValueError("math domain error");
      }
      return result;
    } break;
    default: {
      THROW_PY_TypeError("unsupported operand type(s) for %: '", lhs.type_name(), "' and 'int'");
    } break;
  }
  return 0;
}

RTValue ArithOps::floormod(const Any& lhs, const Any& rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::floormod(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::floormod(lhs.value_.data.v_float64, rhs);
    } break;
    default: {
      THROW_PY_TypeError(
          "unsupported operand type(s) for %: '", lhs.type_name(), "' and '", rhs.type_name(), "'");
    } break;
  }
  return 0;
}

bool ArithOps::eq(const Any& lhs, const Any& rhs) {
  if (rhs.type_code() == TypeIndex::kRuntimeFTList ||
      rhs.type_code() == TypeIndex::kRuntimeFTDict || rhs.type_code() == TypeIndex::kRuntimeFTSet) {
    return rhs.AsObjectViewNoCheck<FTObjectBase>()
        .data()
        .generic_call_attr("__eq__", {lhs.As<RTView>()})
        .As<bool>();
  } else if (rhs.type_code() == TypeIndex::kRuntimeUserData) {
    if (lhs.type_code() == TypeIndex::kRuntimeNullptr) {
      return false;
    }
    return rhs.AsObjectViewNoCheck<UserDataRef>()
        .data()
        .generic_call_attr("__eq__", {lhs.As<RTView>()})
        .As<bool>();
  }
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeNullptr: {
      return rhs.type_code() == TypeIndex::kRuntimeNullptr;
    } break;
    case TypeIndex::kRuntimeOpaqueHandle: {
      return rhs.type_code() == TypeIndex::kRuntimeOpaqueHandle &&
             lhs.value_.data.v_handle == rhs.value_.data.v_handle;
    } break;
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::eq(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::eq(lhs.value_.data.v_float64, rhs);
    } break;
    case TypeIndex::kRuntimeString: {
      return (rhs.type_code() == TypeIndex::kRuntimeString) &&
             (lhs.AsNoCheck<string_view>() == rhs.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return (rhs.type_code() == TypeIndex::kRuntimeUnicode) &&
             (lhs.AsNoCheck<unicode_view>() == rhs.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeTuple: {
      return ArithOps::eq(rhs, lhs.AsObjectViewNoCheck<Tuple>().data());
    } break;
    case TypeIndex::kRuntimeList: {
      return ArithOps::eq(rhs, lhs.AsObjectViewNoCheck<List>().data());
    } break;
    case TypeIndex::kRuntimeSet: {
      return ArithOps::eq(rhs, lhs.AsObjectViewNoCheck<Set>().data());
    } break;
    case TypeIndex::kRuntimeDict: {
      return ArithOps::eq(rhs, lhs.AsObjectViewNoCheck<Dict>().data());
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      return lhs.AsObjectViewNoCheck<FTObjectBase>()
          .data()
          .generic_call_attr("__eq__", {rhs.As<RTView>()})
          .As<bool>();
    } break;
    case TypeIndex::kRuntimeNDArray: {
      return ArithOps::eq(rhs, lhs.AsObjectViewNoCheck<NDArray>().data());
    } break;
    case TypeIndex::kRuntimeDataType: {
      return rhs.type_code() == TypeIndex::kRuntimeDataType &&
             lhs.value_.data.v_type.code == rhs.value_.data.v_type.code &&
             lhs.value_.data.v_type.bits == rhs.value_.data.v_type.bits &&
             lhs.value_.data.v_type.lanes == rhs.value_.data.v_type.lanes;
    } break;
    case TypeIndex::kRuntimeUserData: {
      if (rhs.type_code() == TypeIndex::kRuntimeNullptr) {
        return false;
      }
      return lhs.AsObjectViewNoCheck<UserDataRef>()
          .data()
          .generic_call_attr("__eq__", {rhs.As<RTView>()})
          .As<bool>();
    } break;
    case TypeIndex::kRuntimeOpaqueObject: {
      if (rhs.type_code() != TypeIndex::kRuntimeOpaqueObject) {
        return false;
      }
      auto* lhs_ptr = reinterpret_cast<OpaqueObjectNode*>(lhs.value_.data.v_handle);
      auto* rhs_ptr = reinterpret_cast<OpaqueObjectNode*>(rhs.value_.data.v_handle);
      return lhs_ptr->ptr == rhs_ptr->ptr;
    } break;
    case TypeIndex::kRuntimeObjectRValueRefArg: {
      MXTHROW << "TypeError: unequalable type: 'ObjectRValueRefArg'";
      return false;
    } break;
    case TypeIndex::kRuntimePackedFuncHandle: {
      MXTHROW << "TypeError: unequalable type: 'PackedFunc'";
      return false;
    } break;
    case TypeIndex::kRuntimeDLTensorHandle: {
      MXTHROW << "TypeError: unequalable type: 'DLTensorHandle'";
      return false;
    } break;
    case TypeIndex::kRuntimeContext: {
      MXTHROW << "TypeError: unequalable type: 'Context'";
      return false;
    } break;
    case TypeIndex::kMATXByteArray: {
      MXTHROW << "TypeError: unequalable type: 'ByteArray'";
      return false;
    } break;
    default: {
      if (lhs.type_code() != rhs.type_code()) {
        return false;
      }
      if (lhs.type_code() >= 0) {
        return lhs.ptr<Object>() == rhs.ptr<Object>();
      } else {
        return lhs.value_.data.v_int64 == rhs.value_.data.v_int64;
      }
    } break;
  }
  return false;
}

// TODO: fix comparing of lists and tuples
bool ArithOps::gt(const Any& lhs, const Any& rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::gt(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::gt(lhs.value_.data.v_float64, rhs);
    } break;
    case TypeIndex::kRuntimeString: {
      return lhs.AsNoCheck<string_view>() > rhs.As<string_view>();
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return lhs.AsNoCheck<unicode_view>() > rhs.As<unicode_view>();
    } break;
    case TypeIndex::kRuntimeList: {
      return lhs.AsNoCheck<List>() > rhs.As<List>();
    } break;
    case TypeIndex::kRuntimeTuple: {
      return lhs.AsNoCheck<Tuple>() >= rhs.As<Tuple>();
    } break;
    default: {
      THROW_PY_TypeError("TypeError: '>' not supported between instances of '",
                         lhs.type_name(),
                         "' and '",
                         rhs.type_name(),
                         "'");
    } break;
  }
  return false;
}

// TODO: fix comparing of lists and tuples
bool ArithOps::ge(const Any& lhs, const Any& rhs) {
  switch (lhs.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return ArithOps::ge(lhs.value_.data.v_int64, rhs);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ArithOps::ge(lhs.value_.data.v_float64, rhs);
    } break;
    case TypeIndex::kRuntimeString: {
      return lhs.AsNoCheck<string_view>() >= rhs.As<string_view>();
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return lhs.AsNoCheck<unicode_view>() >= rhs.As<unicode_view>();
    } break;
    case TypeIndex::kRuntimeList: {
      return lhs.AsNoCheck<List>() >= rhs.As<List>();
    } break;
    case TypeIndex::kRuntimeTuple: {
      return lhs.AsNoCheck<Tuple>() >= rhs.As<Tuple>();
    } break;
    default: {
      THROW_PY_TypeError("TypeError: '>=' not supported between instances of '",
                         lhs.type_name(),
                         "' and '",
                         rhs.type_name(),
                         "'");
    } break;
  }
  return false;
}

}  // namespace runtime
}  // namespace matxscript
