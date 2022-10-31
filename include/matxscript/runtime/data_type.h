// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the DataType originates from Halide.
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

#include <cstdint>
#include <sstream>
#include <string>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/dlpack.h>
#include <matxscript/runtime/half.h>
#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {
/*!
 * \brief Runtime primitive data type.
 *
 *  This class is a thin wrapper of DLDataType.
 *  We also make use of DataType in compiler to store quick hint
 */
class DataType {
 public:
  /*!
   * \brief Type code for the DataType.
   *
   * DLPack consistency:
   * 1) kInt is consistent with kDLInt
   * 2) kUInt is consistent with kDLUInt
   * 3) kFloat is consistent with kDLFloat
   */
  enum TypeCode {
    kInt = kDLInt,
    kUInt = kDLUInt,
    kFloat = kDLFloat,
    kHandle = 3,
    kBFloat = kDLBfloat,
    kCustomBegin = 129
  };
  /*! \brief default constructor */
  DataType() {
  }
  /*!
   * \brief Constructor
   * \param dtype The DLDataType
   */
  explicit DataType(DLDataType dtype) : data_(dtype) {
  }
  /*!
   * \brief Constructor
   * \param code The type code.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   */
  DataType(int code, int bits, int lanes);
  /*! \return The type code. */
  int code() const {
    return static_cast<int>(data_.code);
  }
  /*! \return number of bits in the data. */
  int bits() const {
    return static_cast<int>(data_.bits);
  }
  /*! \return number of bytes to store each scalar. */
  int bytes() const {
    return (bits() + 7) / 8;
  }
  /*! \return number of lanes in the data. */
  int lanes() const {
    return static_cast<int>(data_.lanes);
  }
  /*! \return whether type is a scalar type. */
  bool is_scalar() const {
    return lanes() == 1;
  }
  /*! \return whether type is a scalar type. */
  bool is_bool() const {
    return code() == DataType::kUInt && bits() == 1;
  }
  /*! \return whether type is a float type. */
  bool is_float() const {
    return code() == DataType::kFloat;
  }
  /*! \return whether type is a float16 type. */
  bool is_float16() const {
    return is_float() && bits() == 16;
  }
  /*! \return whether type is a bfloat16 type. */
  bool is_bfloat16() const {
    return code() == DataType::kBFloat && bits() == 16;
  }
  /*! \return whether type is an int type. */
  bool is_int() const {
    return code() == DataType::kInt;
  }
  /*! \return whether type is an uint type. */
  bool is_uint() const {
    return code() == DataType::kUInt;
  }
  /*! \return whether type is a handle type. */
  bool is_handle() const {
    return code() == DataType::kHandle && !is_void();
  }
  /*! \return whether type is a vector type. */
  bool is_vector() const {
    return lanes() > 1;
  }
  /*! \return whether type is a bool vector type. */
  bool is_vector_bool() const {
    return is_vector() && bits() == 1;
  }
  /*! \return whether type is a Void type. */
  bool is_void() const {
    return code() == DataType::kHandle && bits() == 0 && lanes() == 0;
  }
  /*!
   * \brief Create a new data type by change lanes to a specified value.
   * \param lanes The target number of lanes.
   * \return the result type.
   */
  DataType with_lanes(int lanes) const {
    return DataType(data_.code, data_.bits, lanes);
  }
  /*!
   * \brief Create a new data type by change bits to a specified value.
   * \param bits The target number of bits.
   * \return the result type.
   */
  DataType with_bits(int bits) const {
    return DataType(data_.code, bits, data_.lanes);
  }
  /*!
   * \brief Get the scalar version of the type.
   * \return the result type.
   */
  DataType element_of() const {
    return with_lanes(1);
  }
  /*!
   * \brief Equal comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator==(const DataType& other) const {
    return data_.code == other.data_.code && data_.bits == other.data_.bits &&
           data_.lanes == other.data_.lanes;
  }
  /*!
   * \brief NotEqual comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator!=(const DataType& other) const {
    return !operator==(other);
  }
  /*!
   * \brief Converter to DLDataType
   * \return the result.
   */
  operator DLDataType() const {
    return data_;
  }

  /*!
   * \brief Construct an int type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \return The constructed data type.
   */
  static DataType Int(int bits, int lanes = 1) {
    return DataType(kDLInt, bits, lanes);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType UInt(int bits, int lanes = 1) {
    return DataType(kDLUInt, bits, lanes);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float(int bits, int lanes = 1) {
    return DataType(kDLFloat, bits, lanes);
  }
  /*!
   * \brief Construct a bool type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Bool(int lanes = 1) {
    return DataType::UInt(1, lanes);
  }
  /*!
   * \brief Construct a handle type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Handle(int bits = 64, int lanes = 1) {
    return DataType(kHandle, bits, lanes);
  }
  /*!
   * \brief Construct a Void type.
   * \return The constructed data type.
   */
  static DataType Void() {
    return DataType(kHandle, 0, 0);
  }
  /*!
   * \brief Get the corresponding type of TVMShapeIndex.
   * \return The type of TVM shape index.
   */
  static DataType ShapeIndex();

  static constexpr uint32_t flatten(uint32_t code, uint32_t bits, uint32_t lanes) {
    return (lanes << 16) | (code << 8) | bits;
  }

  static constexpr uint32_t flatten(const DataType& data) {
    return flatten(data.data_.code, data.data_.bits, data.data_.lanes);
  }

  static constexpr uint32_t flatten(const DLDataType& data) {
    return flatten(data.code, data.bits, data.lanes);
  }

  static std::string debug_str(const DLDataType& data) {
    std::ostringstream stream;
    stream << "type_code=" << data.code << " bits=" << data.bits << " lanes=" << data.lanes;
    return stream.str();
  }

  static std::string debug_str(const DataType& data) {
    return debug_str(data.data_);
  }

 private:
  DLDataType data_;
};

class FlattenedDataType {
 public:
  static constexpr uint32_t INT8 = DataType::flatten(kDLInt, 8, 1);
  static constexpr uint32_t INT16 = DataType::flatten(kDLInt, 16, 1);
  static constexpr uint32_t INT32 = DataType::flatten(kDLInt, 32, 1);
  static constexpr uint32_t INT64 = DataType::flatten(kDLInt, 64, 1);
  static constexpr uint32_t FLOAT16 = DataType::flatten(kDLFloat, 16, 1);
  static constexpr uint32_t FLOAT32 = DataType::flatten(kDLFloat, 32, 1);
  static constexpr uint32_t FLOAT64 = DataType::flatten(kDLFloat, 64, 1);
  static constexpr uint32_t UINT8 = DataType::flatten(kDLUInt, 8, 1);
  static constexpr uint32_t UINT16 = DataType::flatten(kDLUInt, 16, 1);
  static constexpr uint32_t BOOL = DataType::flatten(kDLUInt, 1, 1);
};

/*!
 * \brief Get the number of bytes needed in a vector.
 * \param dtype The data type.
 * \return Number of bytes needed.
 */
int GetVectorBytes(DataType dtype);

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(DLDataType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
/*!
 * \brief Check whether two types are equal .
 * \param lhs The left operand.
 * \param rhs The right operand.
 */
inline bool TypeEqual(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

/*!
 * \brief Runtime utility for getting custom type name from code
 * \param type_code Custom type code
 * \return Custom type name
 */
MATX_DLL String GetCustomTypeName(uint8_t type_code);

/*!
 * \brief Runtime utility for checking whether custom type is registered
 * \param type_code Custom type code
 * \return Bool representing whether type is registered
 */
MATX_DLL bool GetCustomTypeRegistered(uint8_t type_code);

/*!
 * \brief Runtime utility for parsing string of the form "custom[<typename>]"
 * \param s String to parse
 * \param scan pointer to parsing pointer, which is scanning across s
 * \return type code of custom type parsed
 */
MATX_DLL uint8_t ParseCustomDatatype(const String& s, const char** scan);

/*!
 * \brief Convert type code to its name
 * \param type_code The type code .
 * \return The name of type code.
 */
const char* DLDataTypeCode2Str(DLDataTypeCode type_code);

/*!
 * \brief convert a string to TVM type.
 * \param s The string to be converted.
 * \return The corresponding tvm type.
 */
DLDataType String2DLDataType(string_view s);

/*!
 * \brief convert a TVM type to string.
 * \param t The type to be converted.
 * \return The corresponding tvm type in string.
 */
String DLDataType2String(DLDataType t);

std::ostream& operator<<(std::ostream& os, DLDataType t);

inline std::ostream& operator<<(std::ostream& os, const DataType& dtype) {  // NOLINT(*)
  return os << dtype.operator DLDataType();
}

}  // namespace runtime
}  // namespace matxscript

#define MATX_NDARRAY_TYPE_SWITCH(dtype, DType, ...)                       \
  switch (::matxscript::runtime::DataType::flatten(dtype)) {              \
    case ::matxscript::runtime::FlattenedDataType::INT8: {                \
      typedef int8_t DType;                                               \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::INT16: {               \
      typedef int16_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::INT32: {               \
      typedef int32_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::INT64: {               \
      typedef int64_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::FLOAT16: {             \
      typedef Half DType;                                                 \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::FLOAT32: {             \
      typedef float DType;                                                \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::FLOAT64: {             \
      typedef double DType;                                               \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::UINT8: {               \
      typedef uint8_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::UINT16: {              \
      typedef uint16_t DType;                                             \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    default: {                                                            \
      MXCHECK(false) << ::matxscript::runtime::DataType::debug_str(dtype) \
                     << " : unsupported ndarray type";                    \
      break;                                                              \
    }                                                                     \
  }

#define MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DType, ...)             \
  switch (::matxscript::runtime::DataType::flatten(dtype)) {              \
    case ::matxscript::runtime::FlattenedDataType::INT8: {                \
      typedef int8_t DType;                                               \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::INT16: {               \
      typedef int16_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::INT32: {               \
      typedef int32_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::INT64: {               \
      typedef int64_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::FLOAT16: {             \
      typedef Half DType;                                                 \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::FLOAT32: {             \
      typedef float DType;                                                \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::FLOAT64: {             \
      typedef double DType;                                               \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::UINT8: {               \
      typedef uint8_t DType;                                              \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::UINT16: {              \
      typedef uint16_t DType;                                             \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    case ::matxscript::runtime::FlattenedDataType::BOOL: {                \
      typedef bool DType;                                                 \
      { __VA_ARGS__ }                                                     \
      break;                                                              \
    }                                                                     \
    default: {                                                            \
      MXCHECK(false) << ::matxscript::runtime::DataType::debug_str(dtype) \
                     << " : unsupported ndarray type";                    \
      break;                                                              \
    }                                                                     \
  }
