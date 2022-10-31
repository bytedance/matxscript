// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/main/Modules/clinic/_randommodule.c.h
 * https://github.com/python/cpython/blob/main/Modules/_randommodule.c
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
/* Random objects */

/* ------------------------------------------------------------------
   The code in this module was based on a download from:
      http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html

   It was modified in 2002 by Raymond Hettinger as follows:

    * the principal computational lines untouched.

    * renamed genrand_res53() to random_random() and wrapped
      in python calling/return code.

    * genrand_int32() and the helper functions, init_genrand()
      and init_by_array(), were declared static, wrapped in
      Python calling/return code.  also, their global data
      references were replaced with structure references.

    * unused functions from the original were deleted.
      new, original C python code was added to implement the
      Random() interface.

   The following are the verbatim comments from the original code:

   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
    products derived from this software without specific prior written
    permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

/* ---------------------------------------------------------------*/

#include <time.h> /* for seeding to current time */
#ifdef MS_WINDOWS
#include <process.h> /* needed for getpid() */
#else
#include <unistd.h> /* needed for getpid() */
#endif
#include <stdint.h>
#include <array>
#include <cmath>
#include <mutex>
#include <random>

#include <matxscript/runtime/builtins_modules/_randommodule.h>
#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>
#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/py_commons/pymacro.h>
#include <matxscript/runtime/py_commons/pytime.h>
#include <matxscript/runtime/runtime_port.h>

/* Period parameters -- These are all magic.  Don't change. */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfU   /* constant vector a */
#define UPPER_MASK 0x80000000U /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffU /* least significant r bits */

namespace matxscript {
namespace runtime {
namespace py_builtins {

typedef struct {
  int index;
  uint32_t state[N];
  std::mutex lock;
} RandomObject;

#define RandomObject_Check(v) (Py_TYPE(v) == &Random_Type)

/*[clinic input]
module _random
class _random.Random "RandomObject *" "&Random_Type"
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=f79898ae7847c321]*/

/* Random methods */

/* generates a random number on [0,0xffffffff]-interval */
static uint32_t genrand_int32(RandomObject* self) {
  uint32_t y;
  static const uint32_t mag01[2] = {0x0U, MATRIX_A};
  /* mag01[x] = x * MATRIX_A  for x=0,1 */
  uint32_t* mt;

  mt = self->state;
  std::lock_guard<std::mutex> guard(self->lock);
  if (self->index >= N) { /* generate N words at one time */
    int kk;

    for (kk = 0; kk < N - M; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
    }
    for (; kk < N - 1; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
    }
    y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];

    self->index = 0;
  }

  y = mt[self->index++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= (y >> 18);
  return y;
}

/* random_random is the function named genrand_res53 in the original code;
 * generates a random number on [0,1) with 53-bit resolution; note that
 * 9007199254740992 == 2**53; I assume they're spelling "/2**53" as
 * multiply-by-reciprocal in the (likely vain) hope that the compiler will
 * optimize the division away at compile-time.  67108864 is 2**26.  In
 * effect, a contains 27 random bits shifted left 26, and b fills in the
 * lower 26 bits of the 53-bit numerator.
 * The original code credited Isaku Wada for this algorithm, 2002/01/09.
 */

/*[clinic input]
_random.Random.random

  self: self(type="RandomObject *")

random() -> x in the interval [0, 1).
[clinic start generated code]*/

double _random_Random_random_impl(RandomObject* self)
/*[clinic end generated code: output=117ff99ee53d755c input=afb2a59cbbb00349]*/
{
  uint32_t a = genrand_int32(self) >> 5, b = genrand_int32(self) >> 6;
  return double((a * 67108864.0 + b) * (1.0 / 9007199254740992.0));
}

/* initializes mt[N] with a seed */
static void init_genrand(RandomObject* self, uint32_t s) {
  int mti;
  uint32_t* mt;

  mt = self->state;
  mt[0] = s;
  for (mti = 1; mti < N; mti++) {
    mt[mti] = (1812433253U * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                                */
    /* 2002/01/09 modified by Makoto Matsumoto                     */
  }
  self->index = mti;
  return;
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
static void init_by_array(RandomObject* self, uint32_t init_key[], size_t key_length) {
  size_t i, j, k; /* was signed in the original code. RDH 12/16/2002 */
  uint32_t* mt;

  mt = self->state;
  init_genrand(self, 19650218U);
  i = 1;
  j = 0;
  k = (N > key_length ? N : key_length);
  for (; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + init_key[j] +
            (uint32_t)j; /* non linear */
    i++;
    j++;
    if (i >= N) {
      mt[0] = mt[N - 1];
      i = 1;
    }
    if (j >= key_length)
      j = 0;
  }
  for (k = N - 1; k; k--) {
    mt[i] =
        (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i; /* non linear */
    i++;
    if (i >= N) {
      mt[0] = mt[N - 1];
      i = 1;
    }
  }

  mt[0] = 0x80000000U; /* MSB is 1; assuring non-zero initial array */
}

/*
 * The rest is Python-specific code, neither part of, nor derived from, the
 * Twister download.
 */

static int random_seed_urandom(RandomObject* self) {
  uint32_t key[N];

  // Python Impl
  // if (_PyOS_URandomNonblock(key, sizeof(key)) < 0) {
  //   return -1;
  // }

  // C++ Impl
  auto* key_buf = reinterpret_cast<std::random_device::result_type*>(key);
  std::random_device dev;
  for (size_t i = 0; i < sizeof(key) / sizeof(std::random_device::result_type); ++i) {
    key_buf[i] = dev();
  }

  init_by_array(self, key, Py_ARRAY_LENGTH(key));
  return 0;
}

static void random_seed_time_pid(RandomObject* self) {
  int64_t now;
  uint32_t key[5];

  now = _PyTime_GetSystemClock();
  key[0] = (uint32_t)(now & 0xffffffffU);
  key[1] = (uint32_t)(now >> 32);

  key[2] = (uint32_t)getpid();

  now = _PyTime_GetMonotonicClock();
  key[3] = (uint32_t)(now & 0xffffffffU);
  key[4] = (uint32_t)(now >> 32);

  init_by_array(self, key, Py_ARRAY_LENGTH(key));
}

// Only support int64 or nullptr
// Compared with python, object_hash and bigint are not supported
static void random_seed(RandomObject* self, int64_t* arg) {
  uint64_t n;
  uint32_t key[2];
  size_t bits, keyused;

  if (arg == NULL) {
    if (random_seed_urandom(self) < 0) {
      /* Reading system entropy failed, fall back on the worst entropy:
         use the current time and process identifier. */
      random_seed_time_pid(self);
    }
    return;
  }

  /* This algorithm relies on the number being unsigned.
   * So: if the arg is a PyLong, use its absolute value.
   * Otherwise use its hash value, cast to unsigned.
   */
  n = std::abs(*arg);

  uint64_t low = n & 0xFFFFFFFF;
  uint64_t high = n >> 32;

  /* Now split n into 32-bit chunks, from the right. */
  bits = high > 0 ? 64 : 32;

  /* Figure out how many 32-bit chunks this gives us. */
  keyused = bits == 0 ? 1 : (bits - 1) / 32 + 1;
  if (keyused == 1) {
    key[0] = low;
  } else {
    key[0] = high;
    key[1] = low;
    if (kIsBigEndian) {
      key[1] = high;
      key[0] = low;
    } else {
      key[0] = high;
      key[1] = low;
    }
  }
  init_by_array(self, key, keyused);
  return;
}

/*[clinic input]
_random.Random.seed

  self: self(type="RandomObject *")
  n: object = None
  /

seed([n]) -> None.

Defaults to use urandom and falls back to a combination
of the current time and the process identifier.
[clinic start generated code]*/

void _random_Random_seed_impl(RandomObject* self, int64_t* n)
/*[clinic end generated code: output=0fad1e16ba883681 input=78d6ef0d52532a54]*/
{
  return random_seed(self, n);
}

/*[clinic input]
_random.Random.getstate

  self: self(type="RandomObject *")

getstate() -> tuple containing the current state.
[clinic start generated code]*/

Tuple _random_Random_getstate_impl(RandomObject* self)
/*[clinic end generated code: output=bf6cef0c092c7180 input=b937a487928c0e89]*/
{
  std::array<int64_t, N + 1> state;
  int i;
  for (i = 0; i < N; i++) {
    state[i] = self->state[i];
  }
  state[i] = self->index;
  return Tuple(state.begin(), state.end());
}

/*[clinic input]
_random.Random.setstate

  self: self(type="RandomObject *")
  state: object
  /

setstate(state) -> None.  Restores generator state.
[clinic start generated code]*/

void _random_Random_setstate(RandomObject* self, const Tuple& state)
/*[clinic end generated code: output=fd1c3cd0037b6681 input=b3b4efbb1bc66af8]*/
{
  int i;
  unsigned long element;
  long index;
  uint32_t new_state[N];

  /* if (!PyTuple_Check(state)) {
    PyErr_SetString(PyExc_TypeError, "state vector must be a tuple");
    return;
  } */
  if (state.size() != N + 1) {
    THROW_PY_ValueError("state vector is the wrong size");
    return;
  }

  for (i = 0; i < N; i++) {
    element = state[i].As<int64_t>();
    if (element == (unsigned long)-1)
      return;
    new_state[i] = (uint32_t)element;
  }

  index = state[i].As<int64_t>();
  if (index == -1)
    return;
  if (index < 0 || index > N) {
    THROW_PY_ValueError("invalid state");
    return;
  }
  self->index = (int)index;
  for (i = 0; i < N; i++)
    self->state[i] = new_state[i];

  return;
}

/*[clinic input]

_random.Random.getrandbits

  self: self(type="RandomObject *")
  k: int
  /

getrandbits(k) -> x.  Generates an int with k random bits.
[clinic start generated code]*/

uint64_t _random_Random_getrandbits_impl(RandomObject* self, int k)
/*[clinic end generated code: output=b402f82a2158887f input=8c0e6396dd176fc0]*/
{
  uint64_t high;
  uint64_t low;

  if (k <= 0) {
    THROW_PY_ValueError("number of bits must be greater than zero");
  }

  if (k <= 32) /* Fast path */ {
    return genrand_int32(self) >> (32 - k);
  }

  if (k > 64) {
    THROW_PY_ValueError("number of bits must be less than 64");
  }

  low = genrand_int32(self);
  high = genrand_int32(self);

  high >>= 64 - k;
  return (high << 32) + low;
}

static const unsigned char BitLengthTable[32] = {0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                                                 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

static int bits_in_digit(uint64_t d) {
  int d_bits = 0;
  while (d >= 32) {
    d_bits += 6;
    d >>= 6;
  }
  d_bits += (int)BitLengthTable[d];
  return d_bits;
}

struct Random {
  static const int64_t VERSION;
  static const double NV_MAGICCONST;
  static constexpr double TWOPI = 2.0 * M_PI;
  static const double LOG4;
  static const double SG_MAGICCONST;
  static const int64_t BPF = 53;  // Number of bits in a float
  static const double RECIP_BPF;

  /* bits_in_digit(d) returns the unique integer k such that 2**(k-1) <= d <
   2**k if d is nonzero, else 0. */

  explicit Random() : gauss_next() {
    random_seed(&random_impl, nullptr);
  }
  explicit Random(int64_t seed) : gauss_next() {
    random_seed(&random_impl, &seed);
  }

  double random() {
    return py_builtins::_random_Random_random_impl(&random_impl);
  }

  int64_t _randbelow(int64_t n) {
    return this->_randbelow_with_getrandbits(n);
  }

  void seed() {
    return py_builtins::_random_Random_seed_impl(&random_impl, nullptr);
  }

  void seed_int(int64_t n) {
    return py_builtins::_random_Random_seed_impl(&random_impl, &n);
  }

  void seed(const Any& a, int64_t version = 2) {
    auto a_code = a.type_code();
    if (a_code != TypeIndex::kRuntimeNullptr) {
      if (a_code != TypeIndex::kRuntimeInteger) {
        MXTHROW << "random.seed only support int";
      }
      this->seed_int(a.As<int64_t>());
    } else {
      this->seed();
    }
    this->gauss_next = None;
  }

  Tuple getstate() {
    auto internalstate = py_builtins::_random_Random_getstate_impl(&random_impl);
    return Tuple::dynamic(VERSION, internalstate, this->gauss_next);
  }

  void setstate(const Tuple& state) {
    int64_t version = state[0].As<int64_t>();
    MXCHECK(version == VERSION) << "version not match";
    this->gauss_next = state[2];
    Tuple internalstate = state[1].As<Tuple>();
    return py_builtins::_random_Random_setstate(&random_impl, internalstate);
  }

  int64_t getrandbits(int64_t k) {
    return _random_Random_getrandbits_impl(&random_impl, k);
  }

  int64_t randrange(const Any& start, const Any& stop = None, const Any& step = RTView(1)) {
    // This code is a bit messy to make it fast for the
    // common case while still doing adequate error checking.
    if (start.type_code() != TypeIndex::kRuntimeInteger) {
      THROW_PY_ValueError("non-integer start for randrange()");
    }
    int64_t istart = start.As<int64_t>();

    if (stop.is_nullptr()) {
      if (istart > 0) {
        return this->_randbelow(istart);
      }
      THROW_PY_ValueError("empty range for randrange()");
    }
    if (stop.type_code() != TypeIndex::kRuntimeInteger) {
      THROW_PY_ValueError("non-integer stop for randrange()");
    }

    // stop argument supplied.
    int64_t istop = stop.As<int64_t>();
    auto width = istop - istart;
    if (ArithOps::eq(step, 1) && width > 0) {
      return istart + this->_randbelow(width);
    }
    if (ArithOps::eq(step, 1)) {
      THROW_PY_ValueError("empty range for randrange() (", istart, ", ", istop, ", ", width, ")");
    }

    // Non-unit step argument supplied.
    if (step.type_code() != TypeIndex::kRuntimeInteger) {
      THROW_PY_ValueError("non-integer step for randrange()");
    }
    int64_t istep = step.As<int64_t>();
    int64_t n;
    if (istep > 0)
      n = (width + istep - 1) / istep;
    else if (istep < 0)
      n = (width + istep + 1) / istep;
    else {
      THROW_PY_ValueError("zero step for randrange()");
    }

    if (n <= 0) {
      THROW_PY_ValueError("empty range for randrange()");
    }

    return istart + istep * this->_randbelow(n);
  }

  int64_t randint(int64_t a, int64_t b) {
    return this->randrange(RTView(a), RTView(b + 1));
  }

  // TODO(maxiandi): impl func: value choice(seq)
  // TODO(maxiandi): impl func: void shuffle(seq)
  // TODO(maxiandi): impl func: List sample(population, k)
  // TODO(maxiandi): impl func: List choices(...)

  template <
      class T1,
      class T2,
      typename = typename std::enable_if<std::is_arithmetic<
          typename std::remove_cv<typename std::remove_reference<T1>::type>::type>::value>::type,
      typename = typename std::enable_if<std::is_arithmetic<
          typename std::remove_cv<typename std::remove_reference<T2>::type>::type>::value>::type>
  double uniform(T1 a, T2 b) {
    return a + (b - a) * this->random();
  }

  double triangular(double low = 0.0, double high = 1.0, const Any& mode = None) {
    auto u = this->random();
    double c;
    if (mode.is_nullptr()) {
      c = 0.5;
    } else {
      c = (mode.As<double>() - low) / (high - low);
    }
    if (std::isinf(c)) {
      return low;
    }

    if (u > c) {
      u = 1.0 - u;
      c = 1.0 - c;
      std::swap(low, high);
    }
    return low + (high - low) * std::sqrt(u * c);
  }

  double normalvariate(double mu, double sigma) {
    double z;
    while (1) {
      auto u1 = random();
      auto u2 = 1.0 - random();
      z = NV_MAGICCONST * (u1 - 0.5) / u2;
      auto zz = z * z / 4.0;
      if (zz <= -std::log(u2)) {
        break;
      }
    }
    return mu + z * sigma;
  }

  double lognormvariate(double mu, double sigma) {
    return std::exp(this->normalvariate(mu, sigma));
  }

  double expovariate(double lambd) {
    return -std::log(1.0 - this->random()) / lambd;
  }

  double vonmisesvariate(double mu, double kappa) {
    if (kappa <= 1e-6) {
      return TWOPI * this->random();
    }

    auto s = 0.5 / kappa;
    auto r = s + std::sqrt(1.0 + s * s);

    double z;
    while (1) {
      auto u1 = this->random();
      z = std::cos(M_PI * u1);

      auto d = z / (r + z);
      auto u2 = this->random();
      if ((u2 < 1.0 - d * d) || (u2 <= (1.0 - d) * std::exp(d))) {
        break;
      }
    }

    auto q = 1.0 / r;
    auto f = (q + z) / (1.0 + q * z);
    auto u3 = this->random();
    double theta;
    if (u3 > 0.5)
      theta = std::fmod((mu + std::acos(f)), TWOPI);
    else
      theta = std::fmod((mu - std::acos(f)), TWOPI);

    return theta;
  }

  double gammavariate(double alpha, double beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
      THROW_PY_ValueError("gammavariate: alpha and beta must be > 0.0");
    }
    if (alpha > 1.0) {
      // Uses R.C.H. Cheng, "The generation of Gamma
      // variables with non-integral shape parameters",
      // Applied Statistics, (1977), 26, No. 1, p71-74

      auto ainv = std::sqrt(2.0 * alpha - 1.0);
      auto bbb = alpha - LOG4;
      auto ccc = alpha + ainv;

      while (1) {
        auto u1 = this->random();
        if (!(1e-7 < u1 && u1 < 0.9999999)) {
          continue;
        }
        auto u2 = 1.0 - this->random();
        auto v = std::log(u1 / (1.0 - u1)) / ainv;
        auto x = alpha * std::exp(v);
        auto z = u1 * u1 * u2;
        auto r = bbb + ccc * v - x;
        if ((r + SG_MAGICCONST - 4.5 * z >= 0.0) || (r >= std::log(z))) {
          return x * beta;
        }
      }

    } else if (alpha == 1.0) {
      // expovariate(1/beta)
      return -std::log(1.0 - this->random()) * beta;
    } else {  // alpha is between 0 and 1 (exclusive)

      // Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
      double x;
      while (1) {
        auto u = this->random();
        auto b = (M_E + alpha) / M_E;
        auto p = b * u;
        if (p <= 1.0) {
          x = std::pow(p, (1.0 / alpha));
        } else {
          x = -std::log((b - p) / alpha);
        }
        auto u1 = this->random();
        if (p > 1.0) {
          if (u1 <= std::pow(x, (alpha - 1.0))) {
            break;
          }
        } else if (u1 <= std::exp(-x)) {
          break;
        }
      }

      return x * beta;
    }
  }

  double gauss(double mu, double sigma) {
    auto z = this->gauss_next;
    this->gauss_next = None;
    if (z.is_nullptr()) {
      auto x2pi = this->random() * TWOPI;
      auto g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
      z = std::cos(x2pi) * g2rad;
      this->gauss_next = std::sin(x2pi) * g2rad;
    }
    return mu + z.As<double>() * sigma;
  }

  double betavariate(double alpha, double beta) {
    auto y = this->gammavariate(alpha, 1.0);
    if (y == 0) {
      return 0.0;
    } else {
      return y / (y + this->gammavariate(beta, 1.0));
    }
  }

  double paretovariate(double alpha) {
    auto u = 1.0 - this->random();
    return 1.0 / std::pow(u, (1.0 / alpha));
  }

  double weibullvariate(double alpha, double beta) {
    auto u = 1.0 - this->random();
    return alpha * std::pow((-std::log(u)), (1.0 / beta));
  }

 protected:
  int64_t _randbelow_with_getrandbits(int64_t n) {
    auto k = bits_in_digit(n);      // don't use (n-1) here because n can be 1
    auto r = this->getrandbits(k);  // 0 <= r < 2**k
    while (r >= n) {
      r = this->getrandbits(k);
    }
    return r;
  }

  int64_t _randbelow_without_getrandbits(int64_t n, int64_t maxsize = (1ULL << BPF)) {
    if (n >= maxsize) {
      MXLOG(WARNING)
          << ("Underlying random() generator does not supply \n"
              "enough bits to choose from a population range this large.\n"
              "To remove the range limitation, add a getrandbits() method.");
      return int(random() * n);
    }
    if (n == 0) {
      THROW_PY_ValueError("Boundary cannot be zero");
    }
    auto rem = maxsize % n;
    auto limit = double(maxsize - rem) / maxsize;  // int(limit * maxsize) % n == 0
    auto r = random();
    while (r >= limit) {
      r = this->random();
    }
    return int(r * maxsize) % n;
  }

  RandomObject random_impl;
  RTValue gauss_next;
};

const int64_t Random::VERSION = 3;
const double Random::NV_MAGICCONST = 4 * std::exp(-0.5) / std::sqrt(2.0);
const double Random::LOG4 = std::log(4.0);
const double Random::SG_MAGICCONST = 1.0 + std::log(4.5);
const double Random::RECIP_BPF = std::pow(2, -BPF);

static Random* GetGlobalDefaultRandomObject() {
  static Random* self = new Random();
  return self;
}

static Random* DEFAULT_RANDOM_OBJECT = GetGlobalDefaultRandomObject();
static std::mutex DEFAULT_RANDOM_OBJECT_MUTEX;
}  // namespace py_builtins

// for codegen
using random_Random = py_builtins::Random;
MATX_REGISTER_NATIVE_OBJECT(random_Random).SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
  switch (args.size()) {
    case 0: {
      return std::make_shared<random_Random>();
    } break;
    case 1: {
      int64_t seed = args[0].As<int64_t>();
      return std::make_shared<random_Random>(seed);
    } break;
    default: {
      MXTHROW << "[random.Random] Expect 0 or 1 arguments but get " << args.size();
      return nullptr;
    } break;
  }
});

double kernel_random_random() {
  return py_builtins::DEFAULT_RANDOM_OBJECT->random();
}

void kernel_random_seed() {
  std::lock_guard<std::mutex> lock(py_builtins::DEFAULT_RANDOM_OBJECT_MUTEX);
  return py_builtins::DEFAULT_RANDOM_OBJECT->seed();
}

void kernel_random_seed_unroll(const Any& n, int64_t version) {
  std::lock_guard<std::mutex> lock(py_builtins::DEFAULT_RANDOM_OBJECT_MUTEX);
  return py_builtins::DEFAULT_RANDOM_OBJECT->seed(n, version);
}

void kernel_random_seed(PyArgs args) {
  switch (args.size()) {
    case 0: {
      return kernel_random_seed();
    } break;
    case 1: {
      return kernel_random_seed_unroll(args[0]);
    } break;
    case 2: {
      return kernel_random_seed_unroll(args[0], args[1].As<int64_t>());
    } break;
    default: {
      MXTHROW << "[random.seed] Expect 0, 1 or 2 arguments but get " << args.size();
      return;
    } break;
  }
}

Tuple kernel_random_getstate() {
  return py_builtins::DEFAULT_RANDOM_OBJECT->getstate();
}

void kernel_random_setstate(const Tuple& state) {
  std::lock_guard<std::mutex> lock(py_builtins::DEFAULT_RANDOM_OBJECT_MUTEX);
  return py_builtins::DEFAULT_RANDOM_OBJECT->setstate(state);
}

int64_t kernel_random_getrandbits(int64_t k) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->getrandbits(k);
}

double kernel_random_uniform(double a, double b) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->uniform(a, b);
}

double kernel_random_triangular() {
  return py_builtins::DEFAULT_RANDOM_OBJECT->triangular();
}

double kernel_random_triangular(PyArgs args) {
  switch (args.size()) {
    case 0: {
      return py_builtins::DEFAULT_RANDOM_OBJECT->triangular();
    } break;
    case 1: {
      return py_builtins::DEFAULT_RANDOM_OBJECT->triangular(args[0].As<double>());
    } break;
    case 2: {
      return py_builtins::DEFAULT_RANDOM_OBJECT->triangular(args[0].As<double>(),
                                                            args[1].As<double>());
    } break;
    case 3: {
      return py_builtins::DEFAULT_RANDOM_OBJECT->triangular(
          args[0].As<double>(), args[1].As<double>(), args[2]);
    } break;
    default: {
      MXTHROW << "[random.triangular] Expect 0-3 arguments but get " << args.size();
      return 0.0;
    } break;
  }
}

int64_t kernel_random_randint(int64_t a, int64_t b) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->randint(a, b);
}

double kernel_random_normalvariate(double mu, double sigma) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->normalvariate(mu, sigma);
}

double kernel_random_lognormvariate(double mu, double sigma) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->lognormvariate(mu, sigma);
}

double kernel_random_expovariate(double lambd) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->expovariate(lambd);
}

double kernel_random_vonmisesvariate(double mu, double kappa) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->vonmisesvariate(mu, kappa);
}

double kernel_random_gammavariate(double alpha, double beta) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->gammavariate(alpha, beta);
}

double kernel_random_gauss(double mu, double sigma) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->gauss(mu, sigma);
}

double kernel_random_betavariate(double alpha, double beta) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->betavariate(alpha, beta);
}

double kernel_random_paretovariate(double alpha) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->paretovariate(alpha);
}

double kernel_random_weibullvariate(double alpha, double beta) {
  return py_builtins::DEFAULT_RANDOM_OBJECT->weibullvariate(alpha, beta);
}

}  // namespace runtime
}  // namespace matxscript
