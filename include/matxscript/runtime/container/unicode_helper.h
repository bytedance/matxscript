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

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container/unicode_view.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/unicodelib/unicode_ops.h>

namespace matxscript {
namespace runtime {

class String;
class Unicode;
class List;
class Tuple;
class Iterator;
class RTView;
class RTValue;
class Any;
class PyArgs;
template <typename T>
class FTList;

struct UnicodeHelper {
  using self_view = unicode_view;
  using size_type = self_view::size_type;
  using value_type = self_view::value_type;

  static constexpr size_type npos = self_view::npos;
  static constexpr auto whitespaces =
      U"\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";

 public:
  /*! \brief Static helper functions */
  static Unicode Concat(self_view lhs, self_view rhs);
  static Unicode Concat(std::initializer_list<self_view> args);
  static Unicode Repeat(self_view sv, int64_t times);
  static Unicode Lower(self_view sv);
  static Unicode Upper(self_view sv);
  static bool IsDigit(self_view sv) noexcept;
  static bool IsAlpha(self_view sv) noexcept;
  static String Encode(self_view sv);
  static bool Contains(self_view sv, self_view str) noexcept;
  static bool Contains(self_view sv, value_type c) noexcept;
  static int64_t PyFind(self_view sv,
                        self_view str,
                        int64_t start = 0,
                        int64_t end = std::numeric_limits<int64_t>::max()) noexcept;
  static Unicode GetItem(self_view sv, int64_t pos);
  static Unicode GetSlice(self_view sv, int64_t b, int64_t e, int64_t step);
  static List Split(self_view sv, self_view sep = unicode_view(), int64_t maxsplit = -1);
  template <typename T>
  static FTList<T> SplitFT(self_view sv, self_view sep = nullptr, int64_t maxsplit = -1);
  static Unicode Join(self_view sv, const Any& iterable);
  static Unicode Join(self_view sv, const Iterator& iter);
  static Unicode Join(self_view sv, const List& list);
  static Unicode Join(self_view sv, const FTList<Unicode>& list);
  static Unicode Replace(self_view sv, self_view old_s, self_view new_s, int64_t count = -1);
  static bool EndsWith(self_view sv,
                       self_view suffix,
                       int64_t start = 0,
                       int64_t end = std::numeric_limits<int64_t>::max()) noexcept;
  static bool EndsWith(self_view sv,
                       const Tuple& suffixes,
                       int64_t start = 0,
                       int64_t end = std::numeric_limits<int64_t>::max());
  static bool EndsWith(self_view sv,
                       const Any& suffix_or_suffixes,
                       int64_t start = 0,
                       int64_t end = std::numeric_limits<int64_t>::max());
  static bool StartsWith(self_view sv,
                         self_view prefix,
                         int64_t start = 0,
                         int64_t end = std::numeric_limits<int64_t>::max()) noexcept;
  static bool StartsWith(self_view sv,
                         const Tuple& prefixes,
                         int64_t start = 0,
                         int64_t end = std::numeric_limits<int64_t>::max());
  static bool StartsWith(self_view sv,
                         const Any& prefix_or_prefixes,
                         int64_t start = 0,
                         int64_t end = std::numeric_limits<int64_t>::max());
  static Unicode LStrip(self_view sv, self_view chars = self_view{});
  static Unicode RStrip(self_view sv, self_view chars = self_view{});
  static Unicode Strip(self_view sv, self_view chars = self_view{});
  static int64_t Count(self_view sv,
                       self_view x,
                       int64_t start = 0,
                       int64_t end = std::numeric_limits<int64_t>::max()) noexcept;
  static Unicode Format(self_view sv, PyArgs args);

  static self_view AsView(const MATXScriptAny* value);
  static self_view AsViewNoCheck(const MATXScriptAny* value) noexcept;
  static MATXScriptAny CopyFrom(MATXScriptAny value);
  static MATXScriptAny CopyFrom(const MATXScriptAny* value);
  static void Destroy(MATXScriptAny* value) noexcept;
};

template <typename T>
FTList<T> UnicodeHelper::SplitFT(self_view sv, self_view sep, int64_t maxsplit) {
  if (maxsplit < 0) {
    maxsplit = std::numeric_limits<int64_t>::max();
  }
  FTList<T> ret;
  ret.reserve(12);
  if (sep.data() == nullptr) {
    auto* data_last = sv.data();
    auto* data_end = sv.data() + sv.size();
    // skip left space
    while (data_last < data_end && py_unicode_isspace(*data_last)) {
      ++data_last;
    }
    auto* data_cur = data_last + 1;
    while (data_cur < data_end) {
      if (py_unicode_isspace(*data_cur)) {
        if (maxsplit > 0) {
          ret.push_back(T(data_last, data_cur - data_last));
          data_last = data_cur + 1;
          // skip consecutive spaces
          while (data_last < data_end && py_unicode_isspace(*data_last)) {
            ++data_last;
          }
          data_cur = data_last + 1;
          --maxsplit;
        } else {
          ret.push_back(T(data_last, data_end - data_last));
          data_last = data_end;
          break;
        }
      } else {
        ++data_cur;
      }
    }
    if (data_last < data_end) {
      ret.push_back(T(data_last, data_end - data_last));
    }
  } else {
    MXCHECK(!sep.empty()) << "ValueError: empty separator";
    size_type end;
    for (size_type start = 0; start < sv.size(); --maxsplit) {
      if (maxsplit > 0 && (end = sv.find(sep, start)) != npos) {
        auto item = sv.substr(start, end - start);
        ret.append(T(item.data(), item.size()));
        start = end + sep.size();
      } else {
        auto item = sv.substr(start);
        ret.append(T(item.data(), item.size()));
        break;
      }
    }
  }
  return ret;
}

}  // namespace runtime
}  // namespace matxscript
