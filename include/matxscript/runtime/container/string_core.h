// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Taken from https://github.com/facebook/folly/blob/master/folly/FBString.h
 *
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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <matxscript/runtime/bytes_hash.h>
#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/jemalloc_helper.h>
#include <matxscript/runtime/runtime_port.h>

MATXSCRIPT_PUSH_WARNING
// Ignore shadowing warnings within this file, so includers can use -Wshadow.
MATXSCRIPT_GNU_DISABLE_WARNING("-Wshadow")

namespace matxscript {
namespace runtime {

// When compiling with ASan, always heap-allocate the string even if
// it would fit in-situ, so that ASan can detect access to the string
// buffer after it has been invalidated (destroyed, resized, etc.).
// Note that this flag doesn't remove support for in-situ strings, as
// that would break ABI-compatibility and wouldn't allow linking code
// compiled with this flag with code compiled without.
#ifdef MATXSCRIPT_SANITIZE_ADDRESS
#define MATXSCRIPT_STRING_DISABLE_SSO true
#else
#define MATXSCRIPT_STRING_DISABLE_SSO false
#endif

namespace string_detail {

template <class InIt, class OutIt>
inline std::pair<InIt, OutIt> copy_n(InIt b,
                                     typename std::iterator_traits<InIt>::difference_type n,
                                     OutIt d) {
  for (; n != 0; --n, ++b, ++d) {
    *d = *b;
  }
  return std::make_pair(b, d);
}

template <class Pod, class T>
inline void podFill(Pod* b, Pod* e, T c) {
  assert(b && e && b <= e);
  constexpr auto kUseMemset = sizeof(T) == 1;
  if /* constexpr */ (kUseMemset) {
    memset(b, c, size_t(e - b));
  } else {
    auto const ee = b + ((e - b) & ~7u);
    for (; b != ee; b += 8) {
      b[0] = c;
      b[1] = c;
      b[2] = c;
      b[3] = c;
      b[4] = c;
      b[5] = c;
      b[6] = c;
      b[7] = c;
    }
    // Leftovers
    for (; b != e; ++b) {
      *b = c;
    }
  }
}

/*
 * Lightly structured memcpy, simplifies copying PODs and introduces
 * some asserts. Unfortunately using this function may cause
 * measurable overhead (presumably because it adjusts from a begin/end
 * convention to a pointer/size convention, so it does some extra
 * arithmetic even though the caller might have done the inverse
 * adaptation outside).
 */
template <class Pod>
inline void podCopy(const Pod* b, const Pod* e, Pod* d) {
  assert(b != nullptr);
  assert(e != nullptr);
  assert(d != nullptr);
  assert(e >= b);
  assert(d >= e || d + (e - b) <= b);
  memcpy(d, b, (e - b) * sizeof(Pod));
}

/*
 * Lightly structured memmove, simplifies copying PODs and introduces
 * some asserts
 */
template <class Pod>
inline void podMove(const Pod* b, const Pod* e, Pod* d) {
  assert(e >= b);
  memmove(d, b, (e - b) * sizeof(*b));
}
}  // namespace string_detail

/*
 * fbstring_core_model is a mock-up type that defines all required
 * signatures of a fbstring core. The fbstring class itself uses such
 * a core object to implement all of the numerous member functions
 * required by the standard.
 *
 * If you want to define a new core, copy the definition below and
 * implement the primitives. Then plug the core into basic_fbstring as
 * a template argument.

template <class Char>
class fbstring_core_model {
 public:
  fbstring_core_model();
  fbstring_core_model(const fbstring_core_model &);
  fbstring_core_model& operator=(const fbstring_core_model &) = delete;
  ~fbstring_core_model();
  // Returns a pointer to string's buffer (currently only contiguous
  // strings are supported). The pointer is guaranteed to be valid
  // until the next call to a non-const member function.
  const Char * data() const;
  // Much like data(), except the string is prepared to support
  // character-level changes. This call is a signal for
  // e.g. reference-counted implementation to fork the data. The
  // pointer is guaranteed to be valid until the next call to a
  // non-const member function.
  Char* mutableData();
  // Returns a pointer to string's buffer and guarantees that a
  // readable '\0' lies right after the buffer. The pointer is
  // guaranteed to be valid until the next call to a non-const member
  // function.
  const Char * c_str() const;
  // Shrinks the string by delta characters. Asserts that delta <=
  // size().
  void shrink(size_t delta);
  // Expands the string by delta characters (i.e. after this call
  // size() will report the old size() plus delta) but without
  // initializing the expanded region. The expanded region is
  // zero-terminated. Returns a pointer to the memory to be
  // initialized (the beginning of the expanded portion). The caller
  // is expected to fill the expanded area appropriately.
  // If expGrowth is true, exponential growth is guaranteed.
  // It is not guaranteed not to reallocate even if size() + delta <
  // capacity(), so all references to the buffer are invalidated.
  Char* expandNoinit(size_t delta, bool expGrowth);
  // Expands the string by one character and sets the last character
  // to c.
  void push_back(Char c);
  // Returns the string's size.
  size_t size() const;
  // Returns the string's capacity, i.e. maximum size that the string
  // can grow to without reallocation. Note that for reference counted
  // strings that's technically a lie - even assigning characters
  // within the existing size would cause a reallocation.
  size_t capacity() const;
  // Returns true if the data underlying the string is actually shared
  // across multiple strings (in a refcounted fashion).
  bool isShared() const;
  // Makes sure that at least minCapacity characters are available for
  // the string without reallocation. For reference-counted strings,
  // it should fork the data even if minCapacity < size().
  void reserve(size_t minCapacity);
};
*/

/**
 * This is the core of the string. The code should work on 32- and
 * 64-bit and both big- and little-endianan architectures with any
 * Char size.
 *
 * The storage is selected as follows (assuming we store one-byte
 * characters on a 64-bit machine): (a) "small" strings between 0 and
 * 23 chars are stored in-situ without allocation (the rightmost byte
 * stores the size); (b) "medium" strings from 24 through 254 chars
 * are stored in malloc-allocated memory that is copied eagerly; (c)
 * "large" strings of 255 chars and above are stored in a similar
 * structure as medium arrays, except that the string is
 * reference-counted and copied lazily. the reference count is
 * allocated right before the character array.
 *
 * The discriminator between these three strategies sits in two
 * bits of the rightmost char of the storage:
 * - If neither is set, then the string is small. Its length is represented by
 *   the lower-order bits on little-endian or the high-order bits on big-endian
 *   of that rightmost character. The value of these six bits is
 *   `maxSmallSize - size`, so this quantity must be subtracted from
 *   `maxSmallSize` to compute the `size` of the string (see `smallSize()`).
 *   This scheme ensures that when `size == `maxSmallSize`, the last byte in the
 *   storage is \0. This way, storage will be a null-terminated sequence of
 *   bytes, even if all 23 bytes of data are used on a 64-bit architecture.
 *   This enables `c_str()` and `data()` to simply return a pointer to the
 *   storage.
 *
 * - If the MSb is set, the string is medium width.
 *
 * - If the second MSb is set, then the string is large. On little-endian,
 *   these 2 bits are the 2 MSbs of MediumLarge::capacity_, while on
 *   big-endian, these 2 bits are the 2 LSbs. This keeps both little-endian
 *   and big-endian fbstring_core equivalent with merely different ops used
 *   to extract capacity/category.
 */
template <class Char>
class string_core {
  static_assert(sizeof(Char) <= sizeof(uint32_t), "Char size is not supported");

 public:
  struct NoInit {};
  struct Category {
    static constexpr int32_t isSmall = 0;
    static constexpr int32_t isMedium = -1;
    static constexpr int32_t isLarge = -2;
    static constexpr int32_t unknown = INT8_MIN;  // only for string_view
  };

  int32_t category() const noexcept {
    if (category_or_small_len_ >= 0) {
      return Category::isSmall;
    }
    return category_or_small_len_;
  }

  static void DestroyCHost(MATXScriptAny* value) noexcept {
    if (value->pad == Category::isMedium) {
      MediumBuffer::Destroy((Char*)(value->data.v_str_store.v_ml.chars));
      value->code = TypeIndex::kRuntimeNullptr;
    } else if (value->pad == Category::isLarge) {
      RefCounted::decrementRefs((Char*)(value->data.v_str_store.v_ml.chars));
      value->code = TypeIndex::kRuntimeNullptr;
    } else {
      // do nothing
    }
  }

  static string_core MoveFromCHost(MATXScriptAny* value) noexcept {
    string_core result;
    if (value->pad == Category::isMedium) {
      result.c_ml_ = value->data.v_str_store.v_ml;
      result.category_or_small_len_ = Category::isMedium;
      value->code = TypeIndex::kRuntimeNullptr;
    } else if (value->pad == Category::isLarge) {
      result.c_ml_ = value->data.v_str_store.v_ml;
      result.category_or_small_len_ = Category::isLarge;
      value->code = TypeIndex::kRuntimeNullptr;
    } else {
      assert((value->pad >= 0) && (value->pad <= maxSmallSize));
      memcpy(result.bytes_, value->data.v_str_store.v_small_bytes, value->pad * sizeof(Char));
      result.category_or_small_len_ = value->pad;
      value->code = TypeIndex::kRuntimeNullptr;
    }
    return result;
  }

  string_core(const Char* const data, size_t size, int32_t category) {
    if (size <= maxSmallSize) {
      initSmall(data, size);
    } else if (size <= maxMediumSize) {
      initMedium(data, size);
    } else {
      if (category == Category::isLarge) {
        ml_.data_ = (Char*)data;
        ml_.size_ = size;
        RefCounted::incrementRefs(ml_.data_);
        category_or_small_len_ = Category::isLarge;
      } else {
        initLarge(data, size);
      }
    }
    assert(this->size() == size);
    assert(size == 0 || memcmp(this->data(), data, size * sizeof(Char)) == 0);
  }

  void CopyTo(MATXScriptStringStorage* c_store, int32_t* category_or_small_len) const noexcept {
    c_store->v_ml = c_ml_;
    *category_or_small_len = category_or_small_len_;
  }

  void MoveTo(MATXScriptStringStorage* c_store, int32_t* category_or_small_len) noexcept {
    CopyTo(c_store, category_or_small_len);
    reset();
  }

 public:
  string_core() noexcept {
    reset();
  }

  string_core(const string_core& rhs) {
    assert(&rhs != this);
    switch (rhs.category()) {
      case Category::isSmall:
        copySmall(rhs);
        break;
      case Category::isMedium:
        copyMedium(rhs);
        break;
      case Category::isLarge:
        copyLarge(rhs);
        break;
      default:
        ::matxscript::runtime::assume_unreachable();
    }
    assert(size() == rhs.size());
    assert(memcmp(data(), rhs.data(), size() * sizeof(Char)) == 0);
  }

  string_core& operator=(const string_core& rhs) = delete;

  string_core(string_core&& goner) noexcept {
    // Take goner's guts
    ml_ = goner.ml_;
    category_or_small_len_ = goner.category_or_small_len_;
    // Clean goner's carcass
    goner.reset();
  }

  string_core(const Char* const data, const size_t size) {
    if (!MATXSCRIPT_STRING_DISABLE_SSO && size <= maxSmallSize) {
      initSmall(data, size);
    } else if (size <= maxMediumSize) {
      initMedium(data, size);
    } else {
      initLarge(data, size);
    }
    assert(this->size() == size);
    assert(size == 0 || memcmp(this->data(), data, size * sizeof(Char)) == 0);
  }

  string_core(const size_t size, NoInit) {
    if (!MATXSCRIPT_STRING_DISABLE_SSO && size <= maxSmallSize) {
      initSmallNoFill(size);
    } else if (size <= maxMediumSize) {
      initMediumNoFill(size);
    } else {
      initLargeNoFill(size);
    }
    assert(this->size() == size);
  }

  string_core(const size_t size, Char c) {
    Char* d_ptr = nullptr;
    if (!MATXSCRIPT_STRING_DISABLE_SSO && size <= maxSmallSize) {
      d_ptr = initSmallNoFill(size);
    } else if (size <= maxMediumSize) {
      d_ptr = initMediumNoFill(size);
    } else {
      d_ptr = initLargeNoFill(size);
    }
    assert(this->size() == size);
    string_detail::podFill(d_ptr, d_ptr + size, c);
  }

  ~string_core() noexcept {
    if (category() == Category::isSmall) {
      return;
    }
    destroyMediumLarge();
  }

  // swap below doesn't test whether &rhs == this (and instead
  // potentially does extra work) on the premise that the rarity of
  // that situation actually makes the check more expensive than is
  // worth.
  void swap(string_core& rhs) noexcept {
    auto const t = ml_;
    auto const cat_or_len = category_or_small_len_;
    ml_ = rhs.ml_;
    category_or_small_len_ = rhs.category_or_small_len_;
    rhs.ml_ = t;
    rhs.category_or_small_len_ = cat_or_len;
  }

  // In C++11 data() and c_str() are 100% equivalent.
  const Char* data() const noexcept {
    return c_str();
  }

  Char* data() noexcept {
    return c_str();
  }

  Char* mutableData() {
    switch (category()) {
      case Category::isSmall:
        return small_;
      case Category::isMedium:
        return ml_.data_;
      case Category::isLarge:
        return mutableDataLarge();
    }
    ::matxscript::runtime::assume_unreachable();
    return ml_.data_;
  }

  const Char* c_str() const noexcept {
    const Char* ptr = ml_.data_;
    // With this syntax, GCC and Clang generate a CMOV instead of a branch.
    ptr = (category() == Category::isSmall) ? small_ : ptr;
    return ptr;
  }

  void shrink(size_t delta) {
    if (category() == Category::isSmall) {
      shrinkSmall(delta);
    } else if (category() == Category::isMedium || RefCounted::refs(ml_.data_) == 1) {
      shrinkMedium(delta);
    } else {
      shrinkLarge(delta);
    }
  }

  MATXSCRIPT_NO_INLINE
  void reserve(size_t minCapacity) {
    switch (category()) {
      case Category::isSmall:
        reserveSmall(minCapacity);
        break;
      case Category::isMedium:
        reserveMedium(minCapacity);
        break;
      case Category::isLarge:
        reserveLarge(minCapacity);
        break;
      default:
        ::matxscript::runtime::assume_unreachable();
    }
    assert(capacity() >= minCapacity);
  }

  Char* expandNoinit(size_t delta, bool expGrowth = false);

  void push_back(Char c) {
    *expandNoinit(1, /* expGrowth = */ true) = c;
  }

  size_t size() const noexcept {
    size_t ret = ml_.size_;
    ret = (category() == Category::isSmall) ? smallSize() : ret;
    return ret;
  }

  size_t capacity() const noexcept {
    switch (category()) {
      case Category::isSmall:
        return maxSmallSize;
      case Category::isLarge:
        // For large-sized strings, a multi-referenced chunk has no
        // available capacity. This is because any attempt to append
        // data would trigger a new allocation.
        if (RefCounted::refs(ml_.data_) > 1) {
          return ml_.size_;
        } else {
          return RefCounted::capacity(ml_.data_);
        }
        break;
      case Category::isMedium: {
        return MediumBuffer::capacity(ml_.data_);
      } break;
      default:
        ::matxscript::runtime::assume_unreachable();
        break;
    }
    return ml_.size_;
  }

  bool isShared() const noexcept {
    return category() == Category::isLarge && RefCounted::refs(ml_.data_) > 1;
  }

 private:
  Char* c_str() noexcept {
    Char* ptr = ml_.data_;
    // With this syntax, GCC and Clang generate a CMOV instead of a branch.
    ptr = (category() == Category::isSmall) ? small_ : ptr;
    return ptr;
  }

  void reset() noexcept {
    setSmallSize(0);
  }

  MATXSCRIPT_NO_INLINE void destroyMediumLarge() noexcept {
    auto const c = category();
    assert(c != Category::isSmall);
    if (c == Category::isMedium) {
      MediumBuffer::Destroy(ml_.data_);
    } else {
      RefCounted::decrementRefs(ml_.data_);
    }
  }

  struct MediumBuffer {
    size_t capacity_;
    Char data_[1];
    constexpr static size_t getDataOffset() noexcept {
      return offsetof(MediumBuffer, data_);
    }

    static MediumBuffer* fromData(Char* p) noexcept {
      return static_cast<MediumBuffer*>(
          static_cast<void*>(static_cast<unsigned char*>(static_cast<void*>(p)) - getDataOffset()));
    }

    static size_t capacity(Char* p) noexcept {
      return fromData(p)->capacity_;
    }

    static MediumBuffer* create(size_t* size) {
      const size_t allocSize = goodMallocSize(getDataOffset() + (*size + 1) * sizeof(Char));
      auto result = static_cast<MediumBuffer*>(checkedMalloc(allocSize));
      result->capacity_ = (allocSize - getDataOffset()) / sizeof(Char) - 1;
      *size = result->capacity_;
      return result;
    }

    static MediumBuffer* create(const Char* data, size_t* size) {
      const size_t effectiveSize = *size;
      auto result = create(size);
      if (MATXSCRIPT_LIKELY(effectiveSize > 0)) {
        string_detail::podCopy(data, data + effectiveSize, result->data_);
      }
      return result;
    }

    static MediumBuffer* reallocate(Char* const data,
                                    const size_t currentSize,
                                    const size_t currentCapacity,
                                    size_t* newCapacity) {
      assert(*newCapacity > 0 && *newCapacity > currentSize);
      const size_t allocNewCapacity =
          goodMallocSize(getDataOffset() + (*newCapacity + 1) * sizeof(Char));
      auto const dis = fromData(data);
      auto result = static_cast<MediumBuffer*>(
          smartRealloc(dis,
                       getDataOffset() + (currentSize + 1) * sizeof(Char),
                       getDataOffset() + (currentCapacity + 1) * sizeof(Char),
                       allocNewCapacity));
      result->capacity_ = (allocNewCapacity - getDataOffset()) / sizeof(Char) - 1;
      *newCapacity = result->capacity_;
      return result;
    }

    static void Destroy(Char* p) noexcept {
      auto const dis = fromData(p);
      free(dis);
    }
  };

  struct RefCounted {
    std::atomic<size_t> refCount_;
    size_t capacity_;
    Char data_[1];

    constexpr static size_t getDataOffset() noexcept {
      return offsetof(RefCounted, data_);
    }

    static RefCounted* fromData(Char* p) noexcept {
      return static_cast<RefCounted*>(
          static_cast<void*>(static_cast<unsigned char*>(static_cast<void*>(p)) - getDataOffset()));
    }

    static size_t capacity(Char* p) noexcept {
      return fromData(p)->capacity_;
    }

    static size_t refs(Char* p) noexcept {
      return fromData(p)->refCount_.load(std::memory_order_acquire);
    }

    static void incrementRefs(Char* p) noexcept {
      fromData(p)->refCount_.fetch_add(1, std::memory_order_acq_rel);
    }

    static void decrementRefs(Char* p) noexcept {
      auto const dis = fromData(p);
      size_t oldcnt = dis->refCount_.fetch_sub(1, std::memory_order_acq_rel);
      assert(oldcnt > 0);
      if (oldcnt == 1) {
        free(dis);
      }
    }

    static RefCounted* create(size_t* size) {
      const size_t allocSize = goodMallocSize(getDataOffset() + (*size + 1) * sizeof(Char));
      auto result = static_cast<RefCounted*>(checkedMalloc(allocSize));
      result->refCount_.store(1, std::memory_order_release);
      result->capacity_ = (allocSize - getDataOffset()) / sizeof(Char) - 1;
      *size = result->capacity_;
      return result;
    }

    static RefCounted* create(const Char* data, size_t* size) {
      const size_t effectiveSize = *size;
      auto result = create(size);
      if (MATXSCRIPT_LIKELY(effectiveSize > 0)) {
        string_detail::podCopy(data, data + effectiveSize, result->data_);
      }
      return result;
    }

    static RefCounted* reallocate(Char* const data,
                                  const size_t currentSize,
                                  const size_t currentCapacity,
                                  size_t* newCapacity) {
      assert(*newCapacity > 0 && *newCapacity > currentSize);
      const size_t allocNewCapacity =
          goodMallocSize(getDataOffset() + (*newCapacity + 1) * sizeof(Char));
      auto const dis = fromData(data);
      assert(dis->refCount_.load(std::memory_order_acquire) == 1);
      auto result = static_cast<RefCounted*>(
          smartRealloc(dis,
                       getDataOffset() + (currentSize + 1) * sizeof(Char),
                       getDataOffset() + (currentCapacity + 1) * sizeof(Char),
                       allocNewCapacity));
      assert(result->refCount_.load(std::memory_order_acquire) == 1);
      result->capacity_ = (allocNewCapacity - getDataOffset()) / sizeof(Char) - 1;
      *newCapacity = result->capacity_;
      return result;
    }
  };

  struct MediumLarge {
    Char* data_;
    size_t size_;
  };

  static_assert(sizeof(MATXScriptStringMediumLarge) == sizeof(MediumLarge),
                "Corrupt memory layout for string_core.");

  union {
    uint8_t bytes_[sizeof(MediumLarge)];  // For accessing the last byte.
    Char small_[sizeof(MediumLarge) / sizeof(Char)];
    MediumLarge ml_;
    MATXScriptStringMediumLarge c_ml_;
  };
  const uint32_t zero_ = 0;            // for c_str
  int32_t category_or_small_len_ = 0;  // small: >= 0; medium : -1; large: -2

  constexpr static size_t lastChar = sizeof(MediumLarge) - 1;
  constexpr static size_t maxSmallSize = sizeof(MediumLarge) / sizeof(Char);
  constexpr static size_t maxMediumSize = 254 / sizeof(Char);

  static_assert(!(sizeof(MediumLarge) % sizeof(Char)), "Corrupt memory layout for string_core.");

  size_t smallSize() const noexcept {
    assert(category() == Category::isSmall);
    return category_or_small_len_;
  }

  void setSmallSize(size_t s) noexcept {
    // Warning: this should work with uninitialized strings too,
    // so don't assume anything about the previous value of
    // small_[maxSmallSize].
    assert(s <= maxSmallSize);
    small_[s] = '\0';
    category_or_small_len_ = s;
    assert(category() == Category::isSmall && size() == s);
  }

  void copySmall(const string_core&);
  void copyMedium(const string_core&);
  void copyLarge(const string_core&);

  void initSmall(const Char* data, size_t size);
  void initMedium(const Char* data, size_t size);
  void initLarge(const Char* data, size_t size);

  Char* initSmallNoFill(size_t size);
  Char* initMediumNoFill(size_t size);
  Char* initLargeNoFill(size_t size);

  void reserveSmall(size_t minCapacity);
  void reserveMedium(size_t minCapacity);
  void reserveLarge(size_t minCapacity);

  void shrinkSmall(size_t delta);
  void shrinkMedium(size_t delta);
  void shrinkLarge(size_t delta);

  void unshare(size_t minCapacity = 0);
  Char* mutableDataLarge();
};

template <class Char>
inline void string_core<Char>::copySmall(const string_core& rhs) {
  static_assert(offsetof(MediumLarge, data_) == 0, "string_core layout failure");
  static_assert(offsetof(MediumLarge, size_) == sizeof(ml_.data_), "string_core layout failure");
  // Just write the whole thing, don't look at details. In
  // particular we need to copy capacity anyway because we want
  // to set the size (don't forget that the last character,
  // which stores a short string's length, is shared with the
  // ml_.capacity field).
  ml_ = rhs.ml_;
  category_or_small_len_ = rhs.size();
  assert(category() == Category::isSmall && this->size() == rhs.size());
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::copyMedium(const string_core& rhs) {
  // Medium strings are copied eagerly. Don't forget to allocate
  // one extra Char for the null terminator.
  size_t effectiveCapacity = rhs.size();
  auto const newMB = MediumBuffer::create(rhs.c_str(), &effectiveCapacity);
  ml_.data_ = newMB->data_;
  ml_.size_ = rhs.size();
  ml_.data_[ml_.size_] = '\0';
  category_or_small_len_ = Category::isMedium;
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::copyLarge(const string_core& rhs) {
  // Large strings are just refcounted
  ml_ = rhs.ml_;
  RefCounted::incrementRefs(ml_.data_);
  category_or_small_len_ = Category::isLarge;
  assert(category() == Category::isLarge && size() == rhs.size());
}

// Small strings are bitblitted
template <class Char>
inline void string_core<Char>::initSmall(const Char* const data, const size_t size) {
  // Layout is: Char* data_, size_t size_, size_t capacity_
  static_assert(
      sizeof(*this) == sizeof(Char*) + sizeof(size_t) + sizeof(uint32_t) + sizeof(int32_t),
      "string_core has unexpected size");
  static_assert(sizeof(Char*) == sizeof(size_t), "string_core size assumption violation");
  // sizeof(size_t) must be a power of 2
  static_assert((sizeof(size_t) & (sizeof(size_t) - 1)) == 0,
                "string_core size assumption violation");

  // If data is aligned, use fast word-wise copying. Otherwise,
  // use conservative memcpy.
  // The word-wise path reads bytes which are outside the range of
  // the string, and makes ASan unhappy, so we disable it when
  // compiling with ASan.
#ifndef MATXSCRIPT_SANITIZE_ADDRESS
  if ((reinterpret_cast<size_t>(data) & (sizeof(size_t) - 1)) == 0) {
    const size_t byteSize = size * sizeof(Char);
    constexpr size_t wordWidth = sizeof(size_t);
    switch ((byteSize + wordWidth - 1) / wordWidth) {  // Number of words.
      case 2:
        ml_.size_ = reinterpret_cast<const size_t*>(data)[1];
      case 1:
        ml_.data_ = *reinterpret_cast<Char**>(const_cast<Char*>(data));
      case 0:
        break;
    }
  } else
#endif
  {
    if (size != 0) {
      string_detail::podCopy(data, data + size, small_);
    }
  }
  setSmallSize(size);
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::initMedium(const Char* const data,
                                                               const size_t size) {
  // Medium strings are allocated normally. Don't forget to
  // allocate one extra Char for the terminating null.
  size_t effectiveCapacity = size;
  auto const newMB = MediumBuffer::create(data, &effectiveCapacity);
  ml_.data_ = newMB->data_;
  ml_.size_ = size;
  ml_.data_[size] = '\0';
  category_or_small_len_ = Category::isMedium;
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::initLarge(const Char* const data,
                                                              const size_t size) {
  // Large strings are allocated differently
  size_t effectiveCapacity = size;
  auto const newRC = RefCounted::create(data, &effectiveCapacity);
  ml_.data_ = newRC->data_;
  ml_.size_ = size;
  ml_.data_[size] = '\0';
  category_or_small_len_ = Category::isLarge;
}

template <class Char>
inline Char* string_core<Char>::initSmallNoFill(const size_t size) {
  setSmallSize(size);
  return small_;
}

template <class Char>
MATXSCRIPT_NO_INLINE inline Char* string_core<Char>::initMediumNoFill(const size_t size) {
  // Medium strings are allocated normally. Don't forget to
  // allocate one extra Char for the terminating null.
  size_t effectiveCapacity = size;
  auto const newMB = MediumBuffer::create(&effectiveCapacity);
  ml_.data_ = newMB->data_;
  ml_.size_ = size;
  ml_.data_[size] = '\0';
  category_or_small_len_ = Category::isMedium;
  return ml_.data_;
}

template <class Char>
MATXSCRIPT_NO_INLINE inline Char* string_core<Char>::initLargeNoFill(const size_t size) {
  // Large strings are allocated differently
  size_t effectiveCapacity = size;
  auto const newRC = RefCounted::create(&effectiveCapacity);
  ml_.data_ = newRC->data_;
  ml_.size_ = size;
  ml_.data_[size] = '\0';
  category_or_small_len_ = Category::isLarge;
  return ml_.data_;
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::unshare(size_t minCapacity) {
  assert(category() == Category::isLarge);
  auto cur_cap = RefCounted::capacity(ml_.data_);
  size_t effectiveCapacity = std::max(minCapacity, cur_cap);
  auto const newRC = RefCounted::create(&effectiveCapacity);
  // If this fails, someone placed the wrong capacity in an
  // fbstring.
  assert(effectiveCapacity >= cur_cap);
  // Also copies terminator.
  string_detail::podCopy(ml_.data_, ml_.data_ + ml_.size_ + 1, newRC->data_);
  RefCounted::decrementRefs(ml_.data_);
  ml_.data_ = newRC->data_;
  category_or_small_len_ = Category::isLarge;
  // size_ remains unchanged.
}

template <class Char>
inline Char* string_core<Char>::mutableDataLarge() {
  assert(category() == Category::isLarge);
  if (RefCounted::refs(ml_.data_) > 1) {  // Ensure unique.
    unshare();
  }
  return ml_.data_;
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::reserveLarge(size_t minCapacity) {
  assert(category() == Category::isLarge);
  if (RefCounted::refs(ml_.data_) > 1) {  // Ensure unique
    // We must make it unique regardless; in-place reallocation is
    // useless if the string is shared. In order to not surprise
    // people, reserve the new block at current capacity or
    // more. That way, a string's capacity never shrinks after a
    // call to reserve.
    unshare(minCapacity);
  } else {
    // String is not shared, so let's try to realloc (if needed)
    if (minCapacity > RefCounted::capacity(ml_.data_)) {
      // Asking for more memory
      auto const newRC = RefCounted::reallocate(
          ml_.data_, ml_.size_, RefCounted::capacity(ml_.data_), &minCapacity);
      ml_.data_ = newRC->data_;
      category_or_small_len_ = Category::isLarge;
    }
    assert(capacity() >= minCapacity);
  }
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::reserveMedium(size_t minCapacity) {
  assert(category() == Category::isMedium);
  // String is not shared
  if (minCapacity <= MediumBuffer::capacity(ml_.data_)) {
    return;  // nothing to do, there's enough room
  }
  if (minCapacity <= maxMediumSize) {
    // Asking for more memory
    auto const newMB = MediumBuffer::reallocate(
        ml_.data_, ml_.size_, MediumBuffer::capacity(ml_.data_), &minCapacity);
    ml_.data_ = newMB->data_;
  } else {
    // Conversion from medium to large string
    string_core nascent;
    // Will recurse to another branch of this function
    nascent.reserve(minCapacity);
    nascent.ml_.size_ = ml_.size_;
    // Also copies terminator.
    string_detail::podCopy(ml_.data_, ml_.data_ + ml_.size_ + 1, nascent.ml_.data_);
    nascent.swap(*this);
    assert(capacity() >= minCapacity);
  }
}

template <class Char>
MATXSCRIPT_NO_INLINE inline void string_core<Char>::reserveSmall(size_t minCapacity) {
  assert(category() == Category::isSmall);
  if ((!MATXSCRIPT_STRING_DISABLE_SSO) && minCapacity <= maxSmallSize) {
    // small
    // Nothing to do, everything stays put
  } else if (minCapacity <= maxMediumSize) {
    // medium
    auto const newMB = MediumBuffer::create(&minCapacity);
    auto const size = smallSize();
    // Also copies terminator.
    string_detail::podCopy(small_, small_ + size + 1, newMB->data_);
    ml_.data_ = newMB->data_;
    ml_.size_ = size;
    category_or_small_len_ = Category::isMedium;
    assert(capacity() >= minCapacity);
  } else {
    // large
    auto const newRC = RefCounted::create(&minCapacity);
    auto const size = smallSize();
    // Also copies terminator.
    string_detail::podCopy(small_, small_ + size + 1, newRC->data_);
    ml_.data_ = newRC->data_;
    ml_.size_ = size;
    category_or_small_len_ = Category::isLarge;
    assert(capacity() >= minCapacity);
  }
}

template <class Char>
inline Char* string_core<Char>::expandNoinit(size_t delta, bool expGrowth /* = false */) {
  // Strategy is simple: make room, then change size
  assert(capacity() >= size());
  size_t sz, newSz;
  if (category() == Category::isSmall) {
    sz = smallSize();
    newSz = sz + delta;
    if (!MATXSCRIPT_STRING_DISABLE_SSO && MATXSCRIPT_LIKELY(newSz <= maxSmallSize)) {
      setSmallSize(newSz);
      return small_ + sz;
    }
    reserveSmall(expGrowth ? std::max(newSz, 2 * maxSmallSize) : newSz);
  } else {
    sz = ml_.size_;
    newSz = sz + delta;
    if (MATXSCRIPT_UNLIKELY(newSz > capacity())) {
      // ensures not shared
      reserve(expGrowth ? std::max(newSz, 1 + capacity() * 3 / 2) : newSz);
    }
  }
  assert(capacity() >= newSz);
  // Category can't be small - we took care of that above
  assert(category() == Category::isMedium || category() == Category::isLarge);
  ml_.size_ = newSz;
  ml_.data_[newSz] = '\0';
  assert(size() == newSz);
  return ml_.data_ + sz;
}

template <class Char>
inline void string_core<Char>::shrinkSmall(size_t delta) {
  // Check for underflow
  assert(delta <= smallSize());
  setSmallSize(smallSize() - delta);
}

template <class Char>
inline void string_core<Char>::shrinkMedium(size_t delta) {
  // Medium strings and unique large strings need no special
  // handling.
  assert(ml_.size_ >= delta);
  ml_.size_ -= delta;
  ml_.data_[ml_.size_] = '\0';
}

template <class Char>
inline void string_core<Char>::shrinkLarge(size_t delta) {
  assert(ml_.size_ >= delta);
  // Shared large string, must make unique. This is because of the
  // durn terminator must be written, which may trample the shared
  // data.
  if (delta) {
    string_core(ml_.data_, ml_.size_ - delta).swap(*this);
  }
  // No need to write the terminator.
}

#undef MATXSCRIPT_STRING_DISABLE_SSO

}  // namespace runtime
}  // namespace matxscript
