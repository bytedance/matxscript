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

#include <cstdint>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {
/*!
 * \brief Namespace for the list of type index.
 * \note Use struct so that we have to use TypeIndex::ENumName to refer to the constant.
 *       Only allow new index, no modification or deletion.
 *       Those less than 0 are all pod types, and those greater than 0 are all objects.
 */
namespace TypeIndex {

/*! \brief pod types. */
static constexpr int32_t kRuntimeUnknown = INT32_MIN;
static constexpr int32_t kRuntimeNullptr = -1;
static constexpr int32_t kRuntimeOpaqueHandle = -2;
static constexpr int32_t kRuntimeInteger = -3;
static constexpr int32_t kRuntimeFloat = -4;

/*! \brief PackedFunc Struct type. */
static constexpr int32_t kMATXByteArray = -5;
static constexpr int32_t kRuntimeDataType = -6;
static constexpr int32_t kRuntimeContext = -7;
static constexpr int32_t kRuntimeDLTensorHandle = -8;
static constexpr int32_t kRuntimePackedFuncHandle = -9;
static constexpr int32_t kRuntimeObjectRValueRefArg = -10;

/*! \brief runtime::String. */
static constexpr int32_t kRuntimeString = -11;
/*! \brief runtime::Unicode. */
static constexpr int32_t kRuntimeUnicode = -12;

/*! \brief Root object type. */
static constexpr int32_t kRoot = 0;
static constexpr int32_t kRuntimeObject = kRoot;
// Standard static index assignments,
// Frontends can take benefit of these constants.
/*! \brief runtime::Module. */
static constexpr int32_t kRuntimeModule = 1;
/*! \brief runtime::NDArray. */
static constexpr int32_t kRuntimeNDArray = 2;
/*! \brief runtime::StringRef. */
static constexpr int32_t kRuntimeStringRef = 3;
/*! \brief There is no UnicodeRef anymore */
// static constexpr int32_t kRuntimeUnicodeRef = 4;
/*! \brief runtime::Array. */
static constexpr int32_t kRuntimeArray = 5;
/*! \brief runtime::Map. */
static constexpr int32_t kRuntimeMap = 6;
/*! \brief runtime::List. */
static constexpr int32_t kRuntimeList = 7;
/*! \brief runtime::Dict. */
static constexpr int32_t kRuntimeDict = 8;
/*! \brief runtime::Set. */
static constexpr int32_t kRuntimeSet = 9;
/*! \brief runtime::Iterator. */
static constexpr int32_t kRuntimeIterator = 10;
/*! \brief runtime::Generator. */
static constexpr int32_t kRuntimeBoolGenerator = 11;
static constexpr int32_t kRuntimeInt32Generator = 12;
static constexpr int32_t kRuntimeInt64Generator = 13;
static constexpr int32_t kRuntimeFloat32Generator = 14;
static constexpr int32_t kRuntimeFloat64Generator = 15;
static constexpr int32_t kRuntimeRTValueGenerator = 16;
/*! \brief runtime::File. */
static constexpr int32_t kRuntimeFile = 17;
static constexpr int32_t kRuntimeTrie = 18;
static constexpr int32_t kRuntimeRegex = 19;
static constexpr int32_t kRuntimeUserData = 20;
static constexpr int32_t kRuntimeFTObjectBase = 21;
static constexpr int32_t kRuntimeTuple = 22;
static constexpr int32_t kRuntimeOpaqueObject = 23;

/*! \brief runtime::FTList. */
static constexpr int32_t kRuntimeFTList = 24;
/*! \brief runtime::FTDict. */
static constexpr int32_t kRuntimeFTDict = 25;
/*! \brief runtime::FTSet. */
static constexpr int32_t kRuntimeFTSet = 26;

/*! \brief runtime::Kwargs. */
static constexpr int32_t kRuntimeKwargs = 40;

// static assignments that may subject to change.
static constexpr int32_t kStaticIndexEnd = 256;
/*! \brief Type index is allocated during runtime. */
static constexpr int32_t kDynamic = kStaticIndexEnd;

template <typename T>
struct type_index_traits {
  static constexpr int32_t value = kRuntimeUnknown;
};

// null
template <>
struct type_index_traits<std::nullptr_t> {
  static constexpr int32_t value = kRuntimeNullptr;
};

// void*
template <>
struct type_index_traits<void*> {
  static constexpr int32_t value = kRuntimeOpaqueHandle;
};

// integer
template <>
struct type_index_traits<int64_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<uint64_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<int32_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<uint32_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<int16_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<uint16_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<int8_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<uint8_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<char16_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<char32_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<wchar_t> {
  static constexpr int32_t value = kRuntimeInteger;
};
template <>
struct type_index_traits<bool> {
  static constexpr int32_t value = kRuntimeInteger;
};

// float
template <>
struct type_index_traits<float> {
  static constexpr int32_t value = kRuntimeFloat;
};
template <>
struct type_index_traits<double> {
  static constexpr int32_t value = kRuntimeFloat;
};

template <>
struct type_index_traits<char*> {
  static constexpr int32_t value = kRuntimeString;
};

template <>
struct type_index_traits<char[]> {
  static constexpr int32_t value = kRuntimeString;
};

template <>
struct type_index_traits<char32_t*> {
  static constexpr int32_t value = kRuntimeUnicode;
};

template <>
struct type_index_traits<char32_t[]> {
  static constexpr int32_t value = kRuntimeUnicode;
};

};  // namespace TypeIndex

MATX_DLL const char* TypeIndex2Str(int32_t type_code);
MATX_DLL int32_t Str2TypeIndex(string_view str);

}  // namespace runtime
}  // namespace matxscript
