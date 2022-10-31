// Copyright 2022 ByteDance Ltd. and/or its affiliates.
// Taken from https://github.com/abseil/abseil-cpp/blob/master/absl/base/config.h
//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// MATXSCRIPT_HAVE_INTRINSIC_INT128
//
// Checks whether the __int128 compiler extension for a 128-bit integral type is
// supported.
//
// Note: __SIZEOF_INT128__ is defined by Clang and GCC when __int128 is
// supported, but we avoid using it in certain cases:
// * On Clang:
//   * Building using Clang for Windows, where the Clang runtime library has
//     128-bit support only on LP64 architectures, but Windows is LLP64.
// * On Nvidia's nvcc:
//   * nvcc also defines __GNUC__ and __SIZEOF_INT128__, but not all versions
//     actually support __int128.
#if defined(__SIZEOF_INT128__)
#if (defined(__clang__) && !defined(_WIN32)) ||           \
    (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 9) || \
    (defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__))
#define MATXSCRIPT_HAVE_INTRINSIC_INT128 1
#elif defined(__CUDACC__)
// __CUDACC_VER__ is a full version number before CUDA 9, and is defined to a
// string explaining that it has been removed starting with CUDA 9. We use
// nested #ifs because there is no short-circuiting in the preprocessor.
// NOTE: `__CUDACC__` could be undefined while `__CUDACC_VER__` is defined.
#if __CUDACC_VER__ >= 70000
#define MATXSCRIPT_HAVE_INTRINSIC_INT128 1
#endif  // __CUDACC_VER__ >= 70000
#endif  // defined(__CUDACC__)
#endif  // MATXSCRIPT_HAVE_INTRINSIC_INT128

// MATXSCRIPT_IS_LITTLE_ENDIAN
// MATXSCRIPT_IS_BIG_ENDIAN
//
// Checks the endianness of the platform.
//
// Notes: uses the built in endian macros provided by GCC (since 4.6) and
// Clang (since 3.2); see
// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html.
// Otherwise, if _WIN32, assume little endian. Otherwise, bail with an error.
#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define MATXSCRIPT_IS_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define MATXSCRIPT_IS_BIG_ENDIAN 1
#elif defined(_WIN32)
#define MATXSCRIPT_IS_LITTLE_ENDIAN 1
#else
#error "absl endian detection needs to be set up for your compiler"
#endif

// MATXSCRIPT_HAVE_BUILTIN()
//
// Checks whether the compiler supports a Clang Feature Checking Macro, and if
// so, checks whether it supports the provided builtin function "x" where x
// is one of the functions noted in
// https://clang.llvm.org/docs/LanguageExtensions.html
//
// Note: Use this macro to avoid an extra level of #ifdef __has_builtin check.
// http://releases.llvm.org/3.3/tools/clang/docs/LanguageExtensions.html
#ifdef __has_builtin
#define MATXSCRIPT_HAVE_BUILTIN(x) __has_builtin(x)
#else
#define MATXSCRIPT_HAVE_BUILTIN(x) 0
#endif

// MATXSCRIPT_PREDICT_TRUE, MATXSCRIPT_PREDICT_FALSE
//
// Enables the compiler to prioritize compilation using static analysis for
// likely paths within a boolean branch.
//
// Example:
//
//   if (MATXSCRIPT_PREDICT_TRUE(expression)) {
//     return result;                        // Faster if more likely
//   } else {
//     return 0;
//   }
//
// Compilers can use the information that a certain branch is not likely to be
// taken (for instance, a CHECK failure) to optimize for the common case in
// the absence of better information (ie. compiling gcc with `-fprofile-arcs`).
//
// Recommendation: Modern CPUs dynamically predict branch execution paths,
// typically with accuracy greater than 97%. As a result, annotating every
// branch in a codebase is likely counterproductive; however, annotating
// specific branches that are both hot and consistently mispredicted is likely
// to yield performance improvements.
#if MATXSCRIPT_HAVE_BUILTIN(__builtin_expect) || (defined(__GNUC__) && !defined(__clang__))
#define MATXSCRIPT_PREDICT_FALSE(x) (__builtin_expect(false || (x), false))
#define MATXSCRIPT_PREDICT_TRUE(x) (__builtin_expect(false || (x), true))
#else
#define MATXSCRIPT_PREDICT_FALSE(x) (x)
#define MATXSCRIPT_PREDICT_TRUE(x) (x)
#endif
