// Copyright 2022 ByteDance Ltd. and/or its affiliates.
// Acknowledgement:
// Taken from https://github.com/pytorch/pytorch/blob/release/1.11/c10/util/Half.h
// with fixes applied:
// - change namespace to matxscript::runtime for fix conflict with pytorch

#pragma once

#include <cstring>
#include <limits>

#include <matxscript/runtime/runtime_port.h>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#endif

#ifdef __SYCL_DEVICE_ONLY__
#include <CL/sycl.hpp>
#endif

MATXSCRIPT_CLANG_DIAGNOSTIC_PUSH()
#if MATXSCRIPT_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
MATXSCRIPT_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace matxscript {
namespace runtime {

/// Constructors

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half::Half(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  x = __half_as_short(__float2half(value));
#elif defined(__SYCL_DEVICE_ONLY__)
  x = sycl::bit_cast<uint16_t>(sycl::half(value));
#else
  x = detail::fp16_ieee_from_fp32_value(value);
#endif
}

/// Implicit conversions

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __half2float(*reinterpret_cast<const __half*>(&x));
#elif defined(__SYCL_DEVICE_ONLY__)
  return float(sycl::bit_cast<sycl::half>(x));
#else
  return detail::fp16_ieee_to_fp32_value(x);
#endif
}

#if defined(__CUDACC__) || defined(__HIPCC__)
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half::Half(const __half& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half::operator __half() const {
  return *reinterpret_cast<const __half*>(&x);
}
#endif

// CUDA intrinsics

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)) || (defined(__clang__) && defined(__CUDA__))
inline __device__ Half __ldg(const Half* ptr) {
  return __ldg(reinterpret_cast<const __half*>(ptr));
}
#endif

/// Arithmetic

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator+(const Half& a, const Half& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator-(const Half& a, const Half& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator*(const Half& a, const Half& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator/(const Half& a, const Half& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator-(const Half& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || defined(__HIP_DEVICE_COMPILE__)
  return __hneg(a);
#else
  return -static_cast<float>(a);
#endif
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half& operator+=(Half& a, const Half& b) {
  a = a + b;
  return a;
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half& operator-=(Half& a, const Half& b) {
  a = a - b;
  return a;
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half& operator*=(Half& a, const Half& b) {
  a = a * b;
  return a;
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half& operator/=(Half& a, const Half& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator+(Half a, float b) {
  return static_cast<float>(a) + b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator-(Half a, float b) {
  return static_cast<float>(a) - b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator*(Half a, float b) {
  return static_cast<float>(a) * b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator/(Half a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator+(float a, Half b) {
  return a + static_cast<float>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator-(float a, Half b) {
  return a - static_cast<float>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator*(float a, Half b) {
  return a * static_cast<float>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float operator/(float a, Half b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE float& operator+=(float& a, const Half& b) {
  return a += static_cast<float>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float& operator-=(float& a, const Half& b) {
  return a -= static_cast<float>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float& operator*=(float& a, const Half& b) {
  return a *= static_cast<float>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE float& operator/=(float& a, const Half& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator+(Half a, double b) {
  return static_cast<double>(a) + b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator-(Half a, double b) {
  return static_cast<double>(a) - b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator*(Half a, double b) {
  return static_cast<double>(a) * b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator/(Half a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator+(double a, Half b) {
  return a + static_cast<double>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator-(double a, Half b) {
  return a - static_cast<double>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator*(double a, Half b) {
  return a * static_cast<double>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE double operator/(double a, Half b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator+(Half a, int b) {
  return a + static_cast<Half>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator-(Half a, int b) {
  return a - static_cast<Half>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator*(Half a, int b) {
  return a * static_cast<Half>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator/(Half a, int b) {
  return a / static_cast<Half>(b);
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator+(int a, Half b) {
  return static_cast<Half>(a) + b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator-(int a, Half b) {
  return static_cast<Half>(a) - b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator*(int a, Half b) {
  return static_cast<Half>(a) * b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator/(int a, Half b) {
  return static_cast<Half>(a) / b;
}

//// Arithmetic with int64_t

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator+(Half a, int64_t b) {
  return a + static_cast<Half>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator-(Half a, int64_t b) {
  return a - static_cast<Half>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator*(Half a, int64_t b) {
  return a * static_cast<Half>(b);
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator/(Half a, int64_t b) {
  return a / static_cast<Half>(b);
}

inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator+(int64_t a, Half b) {
  return static_cast<Half>(a) + b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator-(int64_t a, Half b) {
  return static_cast<Half>(a) - b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator*(int64_t a, Half b) {
  return static_cast<Half>(a) * b;
}
inline MATXSCRIPT_RUNTIME_HOST_DEVICE Half operator/(int64_t a, Half b) {
  return static_cast<Half>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Half to float.

}  // namespace runtime
}  // namespace matxscript

namespace std {

template <>
class numeric_limits<::matxscript::runtime::Half> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;
  static constexpr ::matxscript::runtime::Half min() {
    return ::matxscript::runtime::Half(0x0400, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half lowest() {
    return ::matxscript::runtime::Half(0xFBFF, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half max() {
    return ::matxscript::runtime::Half(0x7BFF, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half epsilon() {
    return ::matxscript::runtime::Half(0x1400, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half round_error() {
    return ::matxscript::runtime::Half(0x3800, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half infinity() {
    return ::matxscript::runtime::Half(0x7C00, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half quiet_NaN() {
    return ::matxscript::runtime::Half(0x7E00, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half signaling_NaN() {
    return ::matxscript::runtime::Half(0x7D00, ::matxscript::runtime::Half::from_bits());
  }
  static constexpr ::matxscript::runtime::Half denorm_min() {
    return ::matxscript::runtime::Half(0x0001, ::matxscript::runtime::Half::from_bits());
  }
};

}  // namespace std

MATXSCRIPT_CLANG_DIAGNOSTIC_POP()
