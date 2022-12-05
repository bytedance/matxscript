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
#include <matxscript/runtime/generic/generic_constructor_funcs.h>

#include <cctype>
#include <cstdint>
#include <type_traits>

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container/_ft_object_base.h>
#include <matxscript/runtime/container/ft_list.h>
#include <matxscript/runtime/container/list_helper.h>
#include <matxscript/runtime/container/ndarray.h>
#include <matxscript/runtime/container/ndarray_helper.h>
#include <matxscript/runtime/container/unicode_view.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/generator/generator_private.h>
#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/regex/regex_pattern.h>
#include <matxscript/runtime/runtime_port.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Kernel_Iterable
 *****************************************************************************/

namespace Kernel_Iterable {
Iterator make(const Any& obj) {
  switch (obj.type_code()) {
    case TypeIndex::kRuntimeIterator: {
      return obj.AsObjectRefNoCheck<Iterator>();
    } break;
    case TypeIndex::kRuntimeList: {
      return obj.AsObjectViewNoCheck<List>().data().iter();
    } break;
    case TypeIndex::kRuntimeSet: {
      return obj.AsObjectViewNoCheck<Set>().data().iter();
    } break;
    case TypeIndex::kRuntimeDict: {
      return obj.AsObjectViewNoCheck<Dict>().data().key_iter();
    } break;
    case TypeIndex::kRuntimeFTList: {
      return obj.AsObjectViewNoCheck<FTObjectBase>()
          .data()
          .generic_call_attr("__iter__", {})
          .As<Iterator>();
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return obj.AsObjectViewNoCheck<FTObjectBase>()
          .data()
          .generic_call_attr("__iter__", {})
          .As<Iterator>();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return obj.AsObjectViewNoCheck<FTObjectBase>()
          .data()
          .generic_call_attr("__iter__", {})
          .As<Iterator>();
    } break;
    case TypeIndex::kRuntimeString: {
      String container = obj.AsNoCheck<String>();
      return container.iter();
    } break;
    case TypeIndex::kRuntimeUnicode: {
      Unicode container = obj.AsNoCheck<Unicode>();
      return container.iter();
    } break;
    case TypeIndex::kRuntimeTuple: {
      return obj.AsObjectViewNoCheck<Tuple>().data().iter();
    } break;
    case TypeIndex::kRuntimeBoolGenerator: {
      return obj.AsObjectViewNoCheck<BoolGenerator>().data().iter();
    } break;
    case TypeIndex::kRuntimeInt32Generator: {
      return obj.AsObjectViewNoCheck<Int32Generator>().data().iter();
    } break;
    case TypeIndex::kRuntimeInt64Generator: {
      return obj.AsObjectViewNoCheck<Int64Generator>().data().iter();
    } break;
    case TypeIndex::kRuntimeFloat32Generator: {
      return obj.AsObjectViewNoCheck<Float32Generator>().data().iter();
    } break;
    case TypeIndex::kRuntimeFloat64Generator: {
      return obj.AsObjectViewNoCheck<Float64Generator>().data().iter();
    } break;
    case TypeIndex::kRuntimeRTValueGenerator: {
      return obj.AsObjectViewNoCheck<RTValueGenerator>().data().iter();
    } break;
    case TypeIndex::kRuntimeNDArray: {
      return obj.AsObjectViewNoCheck<NDArray>().data().iter();
    } break;
    default: {
      MXTHROW << "Type is not iterable: " << obj.type_name();
      return Iterator();
    }
  }
}
}  // namespace Kernel_Iterable

/******************************************************************************
 * Kernel_bool
 *****************************************************************************/

namespace Kernel_bool {
bool make(const Any& obj) {
  switch (obj.type_code()) {
    case TypeIndex::kRuntimeNullptr: {
      return false;
    } break;
    case TypeIndex::kRuntimeInteger: {
      return static_cast<bool>(obj.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      return static_cast<bool>(obj.AsNoCheck<double>());
    } break;
    case TypeIndex::kRuntimeString: {
      return obj.AsNoCheck<string_view>().size() != 0;
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return obj.AsNoCheck<unicode_view>().size() != 0;
    } break;
    case TypeIndex::kRuntimeList: {
      return obj.AsNoCheck<List>().size() != 0;
    } break;
    case TypeIndex::kRuntimeDict: {
      return obj.AsNoCheck<Dict>().size() != 0;
    } break;
    case TypeIndex::kRuntimeSet: {
      return obj.AsNoCheck<Set>().size() != 0;
    } break;
    case TypeIndex::kRuntimeTuple: {
      return obj.AsNoCheck<Tuple>().size() != 0;
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      return obj.AsObjectViewNoCheck<FTObjectBase>()
                 .data()
                 .generic_call_attr("__len__", {})
                 .As<int64_t>() != 0;
    } break;
    default: {
      THROW_PY_TypeError("'bool' doesn't not supported instance of '", obj.type_name(), "'");
      return false;
    }
  }
}
}  // namespace Kernel_bool

/******************************************************************************
 * Kernel_int64_t
 *****************************************************************************/

namespace Kernel_int64_t {
MATXSCRIPT_ALWAYS_INLINE int64_t string_to_int64(const String& input_str, int64_t base) {
  String str = input_str.rstrip();
  size_t length = str.size();
  MXCHECK_GT(length, 0) << "empty str! should be an int-like str";

  // check pattern and move to digit start
  int sign = 1;
  MXCHECK(!((base != 0 && base < 2) || base > 36)) << "int() arg 2 must be >= 2 and <= 36";
  const char* cstr = str.c_str();
  while (*cstr != '\0' && std::isspace(*cstr)) {
    cstr++;
  }
  if (*cstr == '+') {
    ++cstr;
  } else if (*cstr == '-') {
    ++cstr;
    sign = -1;
  }
  if (base == 0) {
    if (cstr[0] != '0') {
      base = 10;
    } else if (cstr[1] == 'x' || cstr[1] == 'X') {
      base = 16;
    } else if (cstr[1] == 'o' || cstr[1] == 'O') {
      base = 8;
    } else if (cstr[1] == 'b' || cstr[1] == 'B') {
      base = 2;
    } else {
      MXTHROW << "invalid literal for int() with base " << base << ": '" << input_str << "'";
    }
  }
  if (cstr[0] == '0' && cstr[1] != '\0' && !std::isdigit(cstr[1])) {
    bool is_specific_valid = (base == 16 && (cstr[1] == 'x' || cstr[1] == 'X')) ||
                             (base == 8 && (cstr[1] == 'o' || cstr[1] == 'O')) ||
                             (base == 2 && (cstr[1] == 'b' || cstr[1] == 'B'));
    MXCHECK(is_specific_valid) << "invalid literal for int() with base " << base << ": '"
                               << input_str << "'";
    cstr += 2;
    // One underscore allowed here.
    if (*cstr == '_') {
      ++cstr;
    }
  }

  MXCHECK(AsciiIsDigit(string_view(cstr)))
      << "invalid literal for int() with base " << base << ": '" << input_str << "'";

  char* end;
  int64_t ret = std::strtoll(cstr, &end, base);
  if (cstr == end) {
    MXTHROW << "invalid literal for int() with base " << base << ": '" << input_str << "'";
  }
  if (errno == ERANGE) {
    MXTHROW << "invalid literal for int() with base " << base << ": '" << input_str << "'";
  }
  return ret * sign;
}

int64_t make(const String& us, int64_t base) {
  return string_to_int64(us, base);
}
int64_t make(const Unicode& us, int64_t base) {
  return string_to_int64(us.encode(), base);
}

int64_t make(const Any& c, int64_t base) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeString: {
      return make(c.AsNoCheck<String>(), base);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return make(UnicodeHelper::Encode(c.AsNoCheck<unicode_view>()), base);
    } break;
    case TypeIndex::kRuntimeInteger: {
      return make(c.value().data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return make(c.value().data.v_float64);
    } break;
    default: {
      MXTHROW << "expected int64_t acceptable object, but receive: " << c.type_name();
      return 0;
    } break;
  }
  return 0;
}
}  // namespace Kernel_int64_t

/******************************************************************************
 * Kernel_double
 *****************************************************************************/

namespace Kernel_double {
MATXSCRIPT_ALWAYS_INLINE double string_to_float64(const String& str) {
  MXCHECK_GT(str.size(), 0) << "empty str! should be a float-like str";
  char* end;
  double ret = std::strtod(str.c_str(), &end);
  if (str.c_str() == end) {
    MXTHROW << "could not convert string to float: '" << str << "'";
  }
  if (errno == ERANGE) {
    MXTHROW << "could not convert string to float: '" << str << "'";
  }
  return ret;
}

double make(const String& us) {
  return string_to_float64(us);
}
double make(const Unicode& us) {
  return string_to_float64(us.encode());
  return 0.0;
}

double make(const Any& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeString: {
      return make(c.AsNoCheck<String>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return make(UnicodeHelper::Encode(c.AsNoCheck<unicode_view>()));
    } break;
    case TypeIndex::kRuntimeInteger: {
      return make(c.value().data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return make(c.value().data.v_float64);
    } break;
    default: {
      MXTHROW << "expected float64 acceptable object, but receive: " << c.type_name();
      return 0;
    } break;
  }
}
}  // namespace Kernel_double

/******************************************************************************
 * Kernel_String
 *****************************************************************************/

namespace Kernel_String {

String make(const Unicode& us, const Unicode& encoding) {
  MXCHECK_EQ(encoding.view(), unicode_view(U"UTF-8"));
  return us.encode();
}

String make(const Any& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeString: {
      return c.AsNoCheck<String>();
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return UnicodeHelper::Encode(c.AsNoCheck<unicode_view>());
    } break;
    default: {
      MXTHROW << "expected bytes acceptable object, but receive: " << c.type_name();
      return String();
    } break;
  }
}

}  // namespace Kernel_String

/******************************************************************************
 * Kernel_Unicode
 *****************************************************************************/

namespace Kernel_Unicode {
Unicode make(int32_t i32) {
  return StringHelper::Decode(string_view(std::to_string(i32)));
}
Unicode make(int64_t value) {
  static constexpr int n_buf_len = std::numeric_limits<std::uint64_t>::digits10 + 13;
  char buffer[n_buf_len];
  auto len = snprintf(buffer, sizeof(buffer), "%lld", value);
  return StringHelper::Decode(string_view(buffer, len));
}
Unicode make(double d64) {
  // TODO (mxd) : fix bug 10.0 -> "10.0"
  char buffer[8 * (1 << sizeof(d64))];
  auto len = snprintf(buffer, sizeof(buffer), "%.16g", d64);
  return StringHelper::Decode(string_view(buffer, len));
}
Unicode make(float d32) {
  // TODO (mxd) : fix bug 10.0 -> "10.0"
  char buffer[8 * (1 << sizeof(d32))];
  auto len = snprintf(buffer, sizeof(buffer), "%.6g", d32);
  return StringHelper::Decode(string_view(buffer, len));
}
Unicode make(const String& bytes) {
  return bytes.decode();
}
Unicode make(const IUserDataSharedViewRoot& c) {
  auto ud = c.ud_ref;
  return make(ud.__str__());
}

Unicode make(const Any& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeString: {
      return c.AsNoCheck<String>().decode();
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return c.AsNoCheck<Unicode>();
    } break;
    case TypeIndex::kRuntimeInteger: {
      return make(c.value().data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return make(c.value().data.v_float64);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud = c.AsObjectViewNoCheck<UserDataRef>().data();
      return ud.__str__();
    } break;
    default: {
      MXTHROW << "expected unicode acceptable object, but receive: " << c.type_name();
      return Unicode();
    } break;
  }
}
}  // namespace Kernel_Unicode

/******************************************************************************
 * Kernel_Dict
 *****************************************************************************/

namespace Kernel_Dict {
Dict make(const Dict& c) {
  Dict r;
  if (c.defined()) {
    auto* dict_node = c.GetDictNode();
    for (auto& value_type : *dict_node) {
      r.emplace(value_type.first, value_type.second);
    }
  }
  return r;
}
Dict make(const Any& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return make(c.AsObjectViewNoCheck<Dict>().data());
    } break;
    default: {
      MXTHROW << "TypeError: dict(...) not support '" << c.type_name() << "'";
      return {};
    } break;
  }
}
}  // namespace Kernel_Dict

/******************************************************************************
 * Kernel_List
 *****************************************************************************/

namespace Kernel_List {
List make(const Iterator& itr) {
  List d;
  if (itr.defined()) {
    auto d_node = d.GetListNode();
    auto itr_node = itr.GetMutableNode();
    int64_t exp_num = itr_node->Distance();
    if (exp_num >= 0) {
      d.reserve(exp_num);
      for (int64_t i = 0; i < exp_num; ++i) {
        d_node->push_back(itr_node->Next());
      }
    } else {
      while (itr_node->HasNext()) {
        d_node->push_back(itr_node->Next());
      }
    }
  }
  return d;
}
List make(const Any& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeList: {
      return make(c.AsObjectViewNoCheck<List>().data());
    } break;
    case TypeIndex::kRuntimeSet: {
      return make(c.AsObjectViewNoCheck<Set>().data());
    } break;
    case TypeIndex::kRuntimeIterator: {
      return make(c.AsObjectViewNoCheck<Iterator>().data());
    } break;
    default: {
      return make(Kernel_Iterable::make(c));
    } break;
  }
}
}  // namespace Kernel_List

/******************************************************************************
 * Kernel_Set
 *****************************************************************************/

namespace Kernel_Set {
Set make(const Iterator& itr) {
  Set d;
  if (itr.defined()) {
    auto d_node = d.GetSetNode();
    auto itr_node = itr.GetMutableNode();
    int64_t exp_num = itr_node->Distance();
    if (exp_num >= 0) {
      d.reserve(exp_num);
      for (int64_t i = 0; i < exp_num; ++i) {
        d_node->emplace(itr_node->Next());
      }
    } else {
      while (itr_node->HasNext()) {
        d_node->emplace(itr_node->Next());
      }
    }
  }
  return d;
}
Set make(const Any& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return make(c.AsObjectViewNoCheck<Set>().data());
    } break;
    case TypeIndex::kRuntimeList: {
      return make(c.AsObjectViewNoCheck<List>().data());
    } break;
    default: {
      return make(Kernel_Iterable::make(c));
    } break;
  }
}
}  // namespace Kernel_Set

/******************************************************************************
 * Kernel_NDArray
 *****************************************************************************/

namespace Kernel_NDArray {

namespace {
MATXSCRIPT_ALWAYS_INLINE void copy_to(void* data, NDArray& arr) {
  auto* dev_api = DeviceAPI::Get(arr->device);
  MATXScriptStreamHandle stream = dev_api->GetCurrentThreadStream(arr->device);
  dev_api->CopyDataFromTo(data,
                          0,
                          const_cast<void*>(arr.RawData()),
                          0,
                          arr.DataSize(),
                          DLDevice{kDLCPU, 0},
                          arr->device,
                          arr->dtype,
                          stream);
  dev_api->CreateEventSync(stream);
}

template <typename T>
NDArray make_from_ft(const FTList<T>& list,
                     const List& shape,
                     const Unicode& dtype_str,
                     const Unicode& ctx_str) {
  NDArray::check_dtype_valid(dtype_str);
  DataType dtype(String2DLDataType(UTF8Encode(dtype_str.view())));
  std::vector<int64_t> arg_shape;
  int64_t element_num = 1;
  if (shape.empty()) {
    MXCHECK(!list.empty()) << "NDArray: empty list and empty shape";
    element_num = list.size();
    arg_shape.push_back(element_num);
  } else {
    List::size_type dim = shape.size();
    arg_shape.reserve(dim);
    for (List::size_type i = 0; i < dim; ++i) {
      arg_shape.push_back(shape[i].As<int64_t>());
      element_num *= arg_shape[i];
    }
    MXCHECK(list.size() == element_num) << "NDArray: list and shape are mismatched";
  }
  // same type: direct copy
  auto arr = NDArray::Empty(arg_shape, dtype, NDArrayHelper::GetDevice(ctx_str));
  if ((std::is_same<T, int64_t>::value && dtype.is_int() && dtype.bits() == 64) ||
      (std::is_same<T, double>::value && dtype.is_float() && dtype.bits() == 64)) {
    FTObjectBaseNode* obj_ptr = const_cast<FTObjectBaseNode*>(list.get());
    auto& vec = static_cast<FTListNode<T>*>(obj_ptr)->data_;
    copy_to(static_cast<T*>(vec.data()), arr);
    return arr;
  }
  if (ctx_str == U"cpu") {
    FTObjectBaseNode* obj_ptr = const_cast<FTObjectBaseNode*>(list.get());
    auto& vec = static_cast<FTListNode<T>*>(obj_ptr)->data_;
    MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
      DT* data = const_cast<DT*>(arr.Data<DT>());
      for (int64_t i = 0; i < element_num; ++i) {
        data[i] = vec[i];
      }
    });
  } else {
    FTObjectBaseNode* obj_ptr = const_cast<FTObjectBaseNode*>(list.get());
    auto& vec = static_cast<FTListNode<T>*>(obj_ptr)->data_;
    MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
      ListHelper::SimpleVec<DT> data(element_num);
      for (int64_t i = 0; i < element_num; ++i) {
        data.push_back(vec[i]);
      }
      copy_to(static_cast<void*>(data.data()), arr);
    });
  }
  return arr;
}

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
NDArray make_from_scalar(T scalar,
                         const List& shape,
                         const Unicode& dtype_str,
                         const Unicode& ctx_str) {
  NDArray::check_dtype_valid(dtype_str);
  DataType dtype(String2DLDataType(UTF8Encode(dtype_str.view())));
  std::vector<int64_t> arg_shape;
  int64_t element_num = 1;
  if (shape.empty()) {
    THROW_PY_ValueError("matx.NDArray(scalar, shape, ...): shape should not be empty");
  }
  List::size_type dim = shape.size();
  arg_shape.reserve(dim);
  for (List::size_type i = 0; i < dim; ++i) {
    arg_shape.push_back(shape[i].As<int64_t>());
    element_num *= arg_shape[i];
  }

  // same type: direct copy
  auto arr = NDArray::Empty(arg_shape, dtype, NDArrayHelper::GetDevice(ctx_str));
  if (ctx_str == U"cpu") {
    MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
      DT* data = const_cast<DT*>(arr.Data<DT>());
      for (int64_t i = 0; i < element_num; ++i) {
        data[i] = DT(scalar);
      }
    });
  } else {
    MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
      ListHelper::SimpleVec<DT> data(element_num);
      for (int64_t i = 0; i < element_num; ++i) {
        data.push_back(scalar);
      }
      copy_to(static_cast<void*>(data.data()), arr);
    });
  }
  return arr;
}
}  // namespace

NDArray make(const Any& list, const List& shape, const Unicode& dtype_str, const Unicode& ctx_str) {
  switch (list.type_code()) {
    case TypeIndex::kRuntimeList: {
      auto view = list.AsObjectViewNoCheck<List>();
      return make(view.data(), shape, dtype_str, ctx_str);
    } break;
    case TypeIndex::kRuntimeInteger: {
      auto scalar = list.AsNoCheck<int64_t>();
      return make(scalar, shape, dtype_str, ctx_str);
    } break;
    case TypeIndex::kRuntimeFloat: {
      auto scalar = list.AsNoCheck<double>();
      return make(scalar, shape, dtype_str, ctx_str);
    } break;
    case TypeIndex::kRuntimeFTList: {
      if (list.IsObjectRef<FTList<int64_t>>()) {
        auto view = list.AsObjectViewNoCheck<FTList<int64_t>>();
        return make(view.data(), shape, dtype_str, ctx_str);
      }
      if (list.IsObjectRef<FTList<double>>()) {
        auto view = list.AsObjectViewNoCheck<FTList<double>>();
        return make(view.data(), shape, dtype_str, ctx_str);
      }
    } break;
  }
  THROW_PY_TypeError(
      "NDArray make method expects int float List FTList[int] FTList[float] as first argument");
  return None.As<NDArray>();
}

NDArray make(const FTList<int64_t>& list,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str) {
  if (list.size() == 1 && !shape.empty()) {
    // broadcast
    return make_from_scalar(list.get_item(0), shape, dtype_str, ctx_str);
  }
  return make_from_ft(list, shape, dtype_str, ctx_str);
}

NDArray make(const FTList<double>& list,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str) {
  if (list.size() == 1 && !shape.empty()) {
    // broadcast
    return make_from_scalar(list.get_item(0), shape, dtype_str, ctx_str);
  }
  return make_from_ft(list, shape, dtype_str, ctx_str);
}

NDArray make(int64_t scalar, const List& shape, const Unicode& dtype_str, const Unicode& ctx_str) {
  return make_from_scalar(scalar, shape, dtype_str, ctx_str);
}

NDArray make(double scalar, const List& shape, const Unicode& dtype_str, const Unicode& ctx_str) {
  return make_from_scalar(scalar, shape, dtype_str, ctx_str);
}

NDArray make(const List& list,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str) {
  if (list.size() == 1 && !shape.empty()) {
    // broadcast
    if (list[0].Is<int64_t>()) {
      auto scalar = list[0].AsNoCheck<int64_t>();
      return make_from_scalar(scalar, shape, dtype_str, ctx_str);
    }
    if (list[0].Is<double>()) {
      auto scalar = list[0].AsNoCheck<double>();
      return make_from_scalar(scalar, shape, dtype_str, ctx_str);
    }
  }
  NDArray::check_dtype_valid(dtype_str);
  DataType dtype(String2DLDataType(UTF8Encode(dtype_str.view())));
  std::vector<int64_t> arg_shape, list_shape;
  int64_t element_num = 1;
  // get arg_shape
  if (!shape.empty()) {
    List::size_type dim = shape.size();
    arg_shape.reserve(dim);
    for (List::size_type i = 0; i < dim; ++i) {
      arg_shape.push_back(shape[i].As<int64_t>());
      element_num *= arg_shape[i];
    }
  }
  // empty list
  if (list.empty()) {
    if (!arg_shape.empty()) {
      return NDArray::Empty(arg_shape, dtype, NDArrayHelper::GetDevice(ctx_str));
    } else {
      MXTHROW << "invalid input: empty list and empty shape";
    }
  }
  // check list_shape and arg_shape
  MXCHECK(ListHelper::FirstShape(list, list_shape)) << "shape of input list is invalid";
  if (arg_shape.empty()) {
    arg_shape = list_shape;
  } else {
    MXCHECK((list_shape.size() == 1 && list_shape[0] == element_num) ||
            (list_shape.size() == arg_shape.size() &&
             std::equal(list_shape.begin(), list_shape.end(), arg_shape.begin())));
  }
  auto arr = NDArray::Empty(arg_shape, dtype, NDArrayHelper::GetDevice(ctx_str));
  if (ctx_str == U"cpu") {
    MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
      auto data = ListHelper::FlatList<DT>(list, list_shape, (DT*)arr.RawData());
      MXCHECK(data != nullptr) << "shape of input list is invalid";
      return arr;
    });
  }

  MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
    auto data = ListHelper::FlatList<DT>(list, list_shape);
    MXCHECK(data != nullptr) << "shape of input list is invalid";
    copy_to(static_cast<void*>(data->data()), arr);
    return arr;
  });
  return {};
}

}  // namespace Kernel_NDArray

/******************************************************************************
 * Kernel_Trie
 *****************************************************************************/

namespace Kernel_Trie {
Trie make(const Dict& d) {
  std::map<string_view, int64_t> dic;
  std::vector<String> ukeys;
  ukeys.reserve(d.size());
  for (auto& kv : d.items()) {
    MXCHECK(kv.first.IsString() || kv.first.IsUnicode())
        << "[KernelTo<Trie>] Expect argument is dict<str, int>, but get key mismatch: "
        << kv.first.type_name();
    MXCHECK(kv.second.type_code() == TypeIndex::kRuntimeInteger)
        << "[KernelTo<Trie>] Expect argument is dict<str, int>, but get value mismatch: "
        << kv.second.type_name();
    int64_t index = kv.second.As<int64_t>();
    if (kv.first.type_code() == TypeIndex::kRuntimeString) {
      dic.emplace(kv.first.AsNoCheck<string_view>(), index);
    } else {
      ukeys.push_back(UTF8Encode(kv.first.AsNoCheck<unicode_view>()));
      dic.emplace(ukeys.back(), index);
    }
  }
  return Trie(dic);
}
}  // namespace Kernel_Trie

}  // namespace runtime
}  // namespace matxscript
