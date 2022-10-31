// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstddef>

#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {

#if defined(__GNUC__)
// This is for checked malloc-like functions (returns non-null pointer
// which cannot alias any outstanding pointer).
#define MATXSCRIPT_MALLOC_CHECKED_MALLOC __attribute__((__returns_nonnull__, __malloc__))
#else
#define MATXSCRIPT_MALLOC_CHECKED_MALLOC
#endif

size_t goodMallocSize(size_t minSize) noexcept;

/**
 * Trivial wrappers around malloc, calloc, realloc that check for allocation
 * failure and throw std::bad_alloc in that case.
 */

void* checkedMalloc(size_t size);
void* checkedCalloc(size_t n, size_t size);
void* checkedRealloc(void* ptr, size_t size);

/**
 * This function tries to reallocate a buffer of which only the first
 * currentSize bytes are used. The problem with using realloc is that
 * if currentSize is relatively small _and_ if realloc decides it
 * needs to move the memory chunk to a new buffer, then realloc ends
 * up copying data that is not used. It's generally not a win to try
 * to hook in to realloc() behavior to avoid copies - at least in
 * jemalloc, realloc() almost always ends up doing a copy, because
 * there is little fragmentation / slack space to take advantage of.
 */
MATXSCRIPT_MALLOC_CHECKED_MALLOC MATXSCRIPT_NO_INLINE void* smartRealloc(
    void* p, const size_t currentSize, const size_t currentCapacity, const size_t newCapacity);

}  // namespace runtime
}  // namespace matxscript
