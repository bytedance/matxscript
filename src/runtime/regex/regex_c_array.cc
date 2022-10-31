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
#include <matxscript/runtime/regex/regex_c_array.h>

#include <cstring>
#include <memory>

#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {
namespace regex {

int c_array_init(c_array_t* array, size_t n, size_t size) {
  array->nelts = 0;
  array->size = size;
  array->nalloc = n;

  array->base = malloc(n * size + MATXSCRIPT_MEMORY_ALIGNMENT - 1);
  if (array->base == nullptr) {
    return 0;
  }
  array->elts = matxscript_memory_align_ptr(array->base, MATXSCRIPT_MEMORY_ALIGNMENT);
  return 1;
}

c_array_t* c_array_create(size_t n, size_t size) {
  c_array_t* a = nullptr;

  a = (c_array_t*)malloc(sizeof(c_array_t));
  if (a == nullptr) {
    return nullptr;
  }

  if (c_array_init(a, n, size) != 1) {
    return nullptr;
  }

  return a;
}

void c_array_destroy(c_array_t* a) {
  if (a) {
    if (a->base) {
      free(a->base);
    }
    free(a);
  }
}

void* c_array_push(c_array_t* a) {
  void* elt = nullptr;
  void* new_elt = nullptr;
  size_t size = 0;

  if (a->nelts >= a->nalloc) {
    /* the array is full */

    size = a->size * a->nalloc;

    new_elt = realloc(a->base, 2 * size + MATXSCRIPT_MEMORY_ALIGNMENT - 1);
    if (new_elt == nullptr) {
      return nullptr;
    }

    a->base = new_elt;
    a->elts = matxscript_memory_align_ptr(a->base, MATXSCRIPT_MEMORY_ALIGNMENT);
    a->nalloc *= 2;
  }

  elt = (unsigned char*)a->elts + a->size * a->nelts;
  a->nelts++;

  return elt;
}

void* c_array_push_n(c_array_t* a, size_t n) {
  void* elt = nullptr;
  void* new_elt = nullptr;
  size_t size = 0;
  size_t nalloc = 0;

  size = n * a->size;

  if (a->nelts + n >= a->nalloc) {
    /* the array is full, allocate a new array */

    nalloc = 2 * ((n >= a->nalloc) ? n : a->nalloc);

    new_elt = realloc(a->base, nalloc * a->size + MATXSCRIPT_MEMORY_ALIGNMENT - 1);
    if (new_elt == nullptr) {
      return nullptr;
    }

    a->base = new_elt;
    a->elts = matxscript_memory_align_ptr(a->base, MATXSCRIPT_MEMORY_ALIGNMENT);
    a->nalloc = nalloc;
  }

  elt = (unsigned char*)a->elts + a->size * a->nelts;
  a->nelts += n;

  return elt;
}

}  // namespace regex
}  // namespace runtime
}  // namespace matxscript
