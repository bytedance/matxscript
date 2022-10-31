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

#include <matxscript/runtime/hash/hash.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <matxscript/runtime/py_commons/pyhash.h>

namespace matxscript {
namespace runtime {

/*!
 * \brief Hash the binary bytes
 * \param bytes The data pointer
 * \param size The size of the bytes.
 * \return the hash value.
 */
inline size_t BytesHash(const void* bytes, size_t size) noexcept {
  return Hasher::Hash(reinterpret_cast<const unsigned char*>(bytes), size);
}

template <class T, size_t = sizeof(T) / sizeof(size_t)>
struct ScalarHash;

template <class T>
struct ScalarHash<T, 0> : public std::unary_function<T, size_t> {
  size_t operator()(T v) const noexcept {
    union {
      T t;
      size_t a;
    } u;
    u.a = 0;
    u.t = v;
    if (u.a == (size_t)-1) {
      u.a = (size_t)-2;
    }
    return u.a;
  }
};

template <class T>
struct ScalarHash<T, 1> : public std::unary_function<T, size_t> {
  size_t operator()(T v) const noexcept {
    union {
      T t;
      size_t a;
    } u;
    u.t = v;
    if (u.a == (size_t)-1) {
      u.a = (size_t)-2;
    }
    return u.a;
  }
};

template <class T>
struct ScalarHash<T, 2> : public std::unary_function<T, size_t> {
  size_t operator()(T v) const noexcept {
    union {
      T t;
      struct {
        size_t a;
        size_t b;
      } s;
    } u;
    u.t = v;
    return BytesHash(&u, sizeof(u));
  }
};

template <class T>
struct ScalarHash<T, 3> : public std::unary_function<T, size_t> {
  size_t operator()(T v) const noexcept {
    union {
      T t;
      struct {
        size_t a;
        size_t b;
        size_t c;
      } s;
    } u;
    u.t = v;
    return BytesHash(&u, sizeof(u));
  }
};

template <class T>
struct ScalarHash<T, 4> : public std::unary_function<T, size_t> {
  size_t operator()(T v) const noexcept {
    union {
      T t;
      struct {
        size_t a;
        size_t b;
        size_t c;
        size_t d;
      } s;
    } u;
    u.t = v;
    return BytesHash(&u, sizeof(u));
  }
};

template <>
struct ScalarHash<double, sizeof(double) / sizeof(size_t)> {
  size_t operator()(double v) const noexcept {
    return py_builtins::_Py_HashDouble(v);
  }
};

template <class T>
struct ScalarHash<T*, sizeof(void*) / sizeof(size_t)> {
  size_t operator()(T* v) const noexcept {
    return py_builtins::_Py_HashPointer(v);
  }
};

}  // namespace runtime
}  // namespace matxscript
