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
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/uchar_util.h>

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
template <typename T>
class FTList;

struct StringHelper {
  using self_view = string_view;
  using size_type = self_view::size_type;
  using value_type = self_view::value_type;

  static constexpr size_type npos = self_view::npos;
  static constexpr auto whitespaces = " \t\v\r\n\f";

 public:
  static String Concat(self_view lhs, self_view rhs);
  static String Concat(std::initializer_list<self_view> args);
  static String Repeat(self_view str, int64_t times);
  static String Lower(self_view sv);
  static String Upper(self_view sv);
  static bool Isdigit(self_view sv) noexcept;
  static bool Isalpha(self_view sv) noexcept;
  static Unicode Decode(self_view sv);
  static bool Contains(self_view lhs, self_view rhs) noexcept;
  static bool Contains(self_view sv, value_type c) noexcept;
  static int64_t GetItem(self_view sv, int64_t pos);
  static String GetSlice(self_view sv, int64_t b, int64_t e, int64_t step = 1);
  static List Split(self_view sv, self_view sep = nullptr, int64_t maxsplit = -1);
  template <typename T>
  static FTList<T> SplitFT(self_view sv, self_view sep = nullptr, int64_t maxsplit = -1);
  static String Join(self_view sv, const Any& iterable);
  static String Join(self_view sv, const Iterator& iter);
  static String Join(self_view sv, const List& list);
  static String Join(self_view sv, const FTList<String>& list);
  static String JoinStringList(self_view sv, std::initializer_list<String> il);
  static String Replace(self_view sv, self_view old_s, self_view new_s, int64_t count = -1);
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
  static String LStrip(self_view sv, self_view chars = self_view{});
  static String RStrip(self_view sv, self_view chars = self_view{});
  static String Strip(self_view sv, self_view chars = self_view{});
  static int64_t Count(self_view sv,
                       self_view x,
                       int64_t start = 0,
                       int64_t end = std::numeric_limits<int64_t>::max()) noexcept;

  static self_view AsView(const MATXScriptAny* value);
  static self_view AsViewNoCheck(const MATXScriptAny* value) noexcept;
  static MATXScriptAny CopyFrom(const MATXScriptAny* value);
  static MATXScriptAny CopyFrom(MATXScriptAny value);
  static void Destroy(MATXScriptAny* value) noexcept;
};

template <typename T>
FTList<T> StringHelper::SplitFT(self_view sv, self_view sep, int64_t maxsplit) {
  if (maxsplit < 0) {
    maxsplit = std::numeric_limits<int64_t>::max();
  }
  FTList<T> ret;
  ret.reserve(12);
  if (sep.data() == nullptr) {
    auto* data_last = sv.data();
    auto* data_end = sv.data() + sv.size();
    // skip left space
    while (data_last < data_end && UCHAR_ISSPACE(*data_last)) {
      ++data_last;
    }
    auto* data_cur = data_last + 1;
    while (data_cur < data_end) {
      if (UCHAR_ISSPACE(*data_cur)) {
        if (maxsplit > 0) {
          ret.push_back(T(data_last, data_cur - data_last));
          data_last = data_cur + 1;
          // skip consecutive spaces
          while (data_last < data_end && UCHAR_ISSPACE(*data_last)) {
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
