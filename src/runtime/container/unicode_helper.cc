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
#include <matxscript/runtime/container/unicode_helper.h>

#include <cstddef>
#include <sstream>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/ft_list.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

Unicode UnicodeHelper::Concat(self_view lhs, self_view rhs) {
  Unicode ret;
  size_type lhs_size = lhs.size();
  size_type rhs_size = rhs.size();
  ret.reserve(lhs_size + rhs_size);
  ret.append(lhs);
  ret.append(rhs);
  return ret;
}

Unicode UnicodeHelper::Concat(std::initializer_list<self_view> args) {
  size_t cap = 0;
  auto itr = args.begin();
  auto itr_end = args.end();
  for (; itr != itr_end; ++itr) {
    cap += itr->size();
  }
  Unicode ret;
  ret.resizeNoInit(cap);
  auto data = (Unicode::pointer)ret.data();
  for (itr = args.begin(); itr != itr_end; ++itr) {
    Unicode::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
  }
  return ret;
}

Unicode UnicodeHelper::Repeat(self_view sv, int64_t times) {
  times = MATXSCRIPT_UNLIKELY(times < 0) ? 0 : times;
  auto result_size = times * sv.length();
  Unicode::ContainerType store(result_size, Unicode::ContainerType::NoInit{});
  auto* data = (Unicode::pointer)store.data();
  auto* src_data = sv.data();
  auto src_size = sv.size();
  for (int64_t i = 0; i < times; ++i) {
    Unicode::traits_type::copy(data, src_data, src_size);
    data += src_size;
  }
  return Unicode(std::move(store));
}

Unicode UnicodeHelper::Lower(self_view sv) {
  return py_unicode_do_lower_optimize(sv);
}

Unicode UnicodeHelper::Upper(self_view sv) {
  return py_unicode_do_upper_optimize(sv);
}

bool UnicodeHelper::IsDigit(self_view sv) noexcept {
  return py_unicode_isdigit(sv);
}

bool UnicodeHelper::IsAlpha(self_view sv) noexcept {
  return py_unicode_isalpha(sv);
}

String UnicodeHelper::Encode(self_view sv) {
  return UTF8Encode(sv.data(), sv.size());
}

bool UnicodeHelper::Contains(self_view sv, self_view str) noexcept {
  return sv.find(str) != npos;
}

bool UnicodeHelper::Contains(self_view sv, value_type c) noexcept {
  return sv.find(c) != npos;
}

int64_t UnicodeHelper::PyFind(self_view sv, self_view str, int64_t start, int64_t end) noexcept {
  end = slice_index_correction(end, sv.length());
  size_type pos = sv.find(str, start);
  if (pos == self_view::npos || pos > end || pos + str.length() > end) {
    return -1;
  }
  return pos;
}

Unicode UnicodeHelper::GetItem(self_view sv, int64_t pos) {
  int64_t len = sv.size();
  MXCHECK((pos >= 0 && pos < len) || (pos < 0 && pos >= -len)) << "ValueError: index overflow";
  pos = slice_index_correction(pos, len);
  return Unicode(1, sv[pos]);
}

Unicode UnicodeHelper::GetSlice(self_view sv, int64_t b, int64_t e, int64_t step) {
  MXCHECK_GT(step, 0) << "Unicode.slice_load step must be gt 0";
  int64_t len = sv.size();
  b = slice_index_correction(b, len);
  e = slice_index_correction(e, len);
  if (e <= b) {
    return Unicode();
  } else {
    if (step == 1) {
      return Unicode(sv.begin() + b, sv.begin() + e);
    } else {
      Unicode new_val;
      auto new_size = (e - b + step - 1) / step;
      new_val.resizeNoInit(new_size);
      auto* data = (Unicode::pointer)new_val.data();
      auto itr_end = sv.begin() + e;
      for (auto itr = sv.begin() + b; itr < itr_end; itr += step) {
        *data++ = *itr;
      }
      return new_val;
    }
  }
}

List UnicodeHelper::Split(self_view sv, self_view sep, int64_t maxsplit) {
  if (maxsplit < 0) {
    maxsplit = std::numeric_limits<int64_t>::max();
  }
  List ret;
  ListNode* ret_node = ret.GetListNode();
  ret_node->reserve(12);
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
          ret_node->emplace_back(Unicode(data_last, data_cur));
          data_last = data_cur + 1;
          // skip consecutive spaces
          while (data_last < data_end && py_unicode_isspace(*data_last)) {
            ++data_last;
          }
          data_cur = data_last + 1;
          --maxsplit;
        } else {
          ret_node->emplace_back(Unicode(data_last, data_end));
          data_last = data_end;
          break;
        }
      } else {
        ++data_cur;
      }
    }
    if (data_last < data_end) {
      ret_node->emplace_back(Unicode(data_last, data_end));
    }
  } else {
    MXCHECK(!sep.empty()) << "ValueError: empty separator";
    size_type end;
    for (size_type start = 0; start < sv.size(); --maxsplit) {
      if (maxsplit > 0 && (end = sv.find(sep, start)) != npos) {
        ret_node->emplace_back(Unicode(sv.substr(start, end - start)));
        start = end + sep.size();
      } else {
        ret_node->emplace_back(Unicode(sv.substr(start)));
        break;
      }
    }
  }
  return ret;
}

Unicode UnicodeHelper::Join(self_view sv, const Any& iterable) {
  switch (iterable.type_code()) {
    case TypeIndex::kRuntimeIterator: {
      return Join(sv, iterable.AsObjectRefNoCheck<Iterator>());
    } break;
    case TypeIndex::kRuntimeList: {
      return Join(sv, iterable.AsObjectRefNoCheck<List>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      return Join(sv, iterable.AsObjectRef<FTList<Unicode>>());
    } break;
    default: {
      return Join(sv, Kernel_Iterable::make(iterable));
    } break;
  }
}

Unicode UnicodeHelper::Join(self_view sv, const Iterator& iter) {
  bool has_next = iter.HasNext();
  if (!has_next) {
    return Unicode();
  }
  Unicode ret{iter.Next(&has_next).As<unicode_view>()};
  while (has_next) {
    ret.append(sv);
    ret.append(iter.Next(&has_next).As<unicode_view>());
  }
  return ret;
}

Unicode UnicodeHelper::Join(self_view sv, const List& list) {
  size_t cap = 0;
  auto itr = list.begin();
  auto itr_end = list.end();
  bool not_empty = itr != itr_end;
  if (not_empty) {
    cap += itr->As<unicode_view>().size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    cap += sv.size();
    cap += itr->As<unicode_view>().size();
  }
  itr = list.begin();
  Unicode ret;
  ret.resizeNoInit(cap);
  Unicode::pointer data = (Unicode::pointer)ret.data();
  if (not_empty) {
    const auto& view = itr->AsNoCheck<unicode_view>();
    Unicode::traits_type::copy(data, view.data(), view.size());
    data += view.size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    Unicode::traits_type::copy(data, sv.data(), sv.size());
    data += sv.size();
    const auto& view = itr->AsNoCheck<unicode_view>();
    Unicode::traits_type::copy(data, view.data(), view.size());
    data += view.size();
  }
  return ret;
}

Unicode UnicodeHelper::Join(self_view sv, const FTList<Unicode>& list) {
  size_t cap = 0;
  auto itr = list.begin();
  auto itr_end = list.end();
  bool not_empty = itr != itr_end;
  if (not_empty) {
    cap += itr->size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    cap += sv.size();
    cap += itr->size();
  }
  itr = list.begin();
  Unicode ret;
  ret.resizeNoInit(cap);
  Unicode::pointer data = (Unicode::pointer)ret.data();
  if (not_empty) {
    Unicode::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    Unicode::traits_type::copy(data, sv.data(), sv.size());
    data += sv.size();
    Unicode::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
  }
  return ret;
}

Unicode UnicodeHelper::Replace(self_view sv, self_view old_s, self_view new_s, int64_t count) {
  if (count < 0) {
    count = std::numeric_limits<int64_t>::max();
  }
  Unicode ret;
  // What's the proper value of the capacity?
  if (new_s.length() > old_s.length()) {
    ret.reserve(sv.size() * 1.5);
  } else {
    ret.reserve(sv.size());
  }
  size_type current = 0, next;
  while ((next = sv.find(old_s, current)) != npos && count > 0) {
    ret.append(self_view(sv.data() + current, next - current));
    ret.append(new_s);
    current = next + old_s.size();
    --count;
  }
  ret.append(self_view(sv.data() + current, sv.size() - current));
  return ret;
}

bool UnicodeHelper::EndsWith(self_view sv, self_view suffix, int64_t start, int64_t end) noexcept {
  // [start:end)
  int64_t suffix_len = suffix.length();
  int64_t len = sv.length();
  start = slice_index_correction(start, len);
  end = slice_index_correction(end, len);

  if (end - start < suffix_len || start >= sv.length()) {
    return false;
  }
  return suffix.compare(sv.SubStrNoCheck(end - suffix_len, suffix_len)) == 0;
}
bool UnicodeHelper::EndsWith(self_view sv, const Tuple& suffixes, int64_t start, int64_t end) {
  for (const auto& suffix : suffixes) {
    if (!suffix.Is<unicode_view>()) {
      THROW_PY_TypeError("tuple for endswith must only contain str, not ", suffix.type_name());
    }
    if (EndsWith(sv, suffix.AsNoCheck<unicode_view>(), start, end)) {
      return true;
    }
  }
  return false;
}
bool UnicodeHelper::EndsWith(self_view sv,
                             const Any& suffix_or_suffixes,
                             int64_t start,
                             int64_t end) {
  switch (suffix_or_suffixes.type_code()) {
    case TypeIndex::kRuntimeUnicode: {
      return EndsWith(sv, suffix_or_suffixes.As<unicode_view>(), start, end);
    } break;
    case TypeIndex::kRuntimeTuple: {
      return EndsWith(sv, suffix_or_suffixes.AsObjectViewNoCheck<Tuple>().data(), start, end);
    } break;
    default: {
      THROW_PY_TypeError("endswith first arg must be str or a tuple of str, not ",
                         suffix_or_suffixes.type_name());
      return false;
    } break;
  }
}

bool UnicodeHelper::StartsWith(self_view sv,
                               self_view prefix,
                               int64_t start,
                               int64_t end) noexcept {
  // [start:end)
  int64_t prefix_len = prefix.length();
  int64_t len = sv.length();
  start = slice_index_correction(start, len);
  end = slice_index_correction(end, len);

  if (end - start < prefix_len || start >= sv.length()) {
    return false;
  }
  return prefix.compare(sv.SubStrNoCheck(start, prefix_len)) == 0;
}
bool UnicodeHelper::StartsWith(self_view sv, const Tuple& prefixes, int64_t start, int64_t end) {
  for (const auto& prefix : prefixes) {
    if (!prefix.Is<unicode_view>()) {
      THROW_PY_TypeError("tuple for startswith must only contain str, not ", prefix.type_name());
    }
    if (StartsWith(sv, prefix.AsNoCheck<unicode_view>(), start, end)) {
      return true;
    }
  }
  return false;
}
bool UnicodeHelper::StartsWith(self_view sv,
                               const Any& prefix_or_prefixes,
                               int64_t start,
                               int64_t end) {
  switch (prefix_or_prefixes.type_code()) {
    case TypeIndex::kRuntimeUnicode: {
      return StartsWith(sv, prefix_or_prefixes.As<unicode_view>(), start, end);
    } break;
    case TypeIndex::kRuntimeTuple: {
      return StartsWith(sv, prefix_or_prefixes.AsObjectViewNoCheck<Tuple>().data(), start, end);
    } break;
    default: {
      THROW_PY_TypeError("startswith first arg must be str or a tuple of str, not ",
                         prefix_or_prefixes.type_name());
      return false;
    } break;
  }
}

Unicode UnicodeHelper::LStrip(self_view sv, self_view chars) {
  auto* data = sv.data();
  auto* data_end = sv.data() + sv.size();
  if (chars.data() == nullptr) {
    while (data < data_end && py_unicode_isspace(*data)) {
      ++data;
    }
    if (data < data_end) {
      return Unicode(data, data_end - data);
    } else {
      return Unicode{};
    }
  } else {
    while (data < data_end && chars.find(*data) != chars.npos) {
      ++data;
    }
    if (data < data_end) {
      return Unicode(data, data_end - data);
    } else {
      return Unicode{};
    }
  }
}

Unicode UnicodeHelper::RStrip(self_view sv, self_view chars) {
  auto* data_begin = sv.data();
  auto* data = data_begin + sv.size() - 1;
  if (chars.data() == nullptr) {
    while (data >= data_begin && py_unicode_isspace(*data)) {
      --data;
    }
    if (data >= data_begin) {
      return Unicode(data_begin, data - data_begin + 1);
    } else {
      return Unicode{};
    }
  } else {
    while (data >= data_begin && chars.find(*data) != chars.npos) {
      --data;
    }
    if (data >= data_begin) {
      return Unicode(data_begin, data - data_begin + 1);
    } else {
      return Unicode{};
    }
  }
}

Unicode UnicodeHelper::Strip(self_view sv, self_view chars) {
  auto* data_begin = sv.data();
  auto* data_end = sv.data() + sv.size();
  auto* data_left = data_begin;
  auto* data_right = data_begin + sv.size() - 1;
  if (chars.data() == nullptr) {
    while (data_left < data_end && py_unicode_isspace(*data_left)) {
      ++data_left;
    }
    while (data_right > data_left && py_unicode_isspace(*data_right)) {
      --data_right;
    }
    if (data_right >= data_left) {
      return Unicode(data_left, data_right - data_left + 1);
    } else {
      return Unicode{};
    }
  } else {
    while (data_left < data_end && chars.find(*data_left) != chars.npos) {
      ++data_left;
    }
    while (data_right > data_left && chars.find(*data_right) != chars.npos) {
      --data_right;
    }
    if (data_right >= data_left) {
      return Unicode(data_left, data_right - data_left + 1);
    } else {
      return Unicode{};
    }
  }
}

int64_t UnicodeHelper::Count(self_view sv, self_view x, int64_t start, int64_t end) noexcept {
  int64_t x_len = x.length();
  int64_t len = sv.length();
  start = slice_index_correction(start, len);
  end = slice_index_correction(end, len);
  int64_t ret = 0;
  while (start < end) {
    start = sv.find(x, start);
    if (start > end || start == npos) {
      break;
    }
    ++ret;
    start += x_len;
  }
  return ret;
}

Unicode UnicodeHelper::Format(self_view sv, PyArgs args) {
  std::stringstream ss;
  bool left_bracket = false;
  bool right_bracket = false;
  auto arg_itr = args.begin();
  for (auto c : UTF8Encode(sv)) {
    if (left_bracket) {
      if (c == U'{') {
        ss << '{';
        left_bracket = false;
      } else if (c == U'}') {
        MXCHECK(arg_itr != args.end()) << "tuple index out of range";
        ss << *arg_itr;
        ++arg_itr;
        left_bracket = false;
      } else {
        MXCHECK(false) << "keyword format string is not supported now";
      }
    } else if (right_bracket) {
      if (c == U'}') {
        ss << '}';
        right_bracket = false;
      } else {
        MXCHECK(false) << "ingle '}' encountered in format string";
      }
    } else {
      if (c == U'{') {
        left_bracket = true;
      } else if (c == U'}') {
        right_bracket = true;
      } else {
        ss << c;
      }
    }
  }
  MXCHECK(!left_bracket) << "Single '{' encountered in format string";
  MXCHECK(!right_bracket) << "Single '}' encountered in format string";
  return UTF8Decode(String(ss.str()));
}

UnicodeHelper::self_view UnicodeHelper::AsViewNoCheck(const MATXScriptAny* value) noexcept {
  if (value->pad >= 0) {
    return self_view{value->data.v_str_store.v_small_chars, size_t(value->pad), 0};
  } else {
    return self_view{
        value->data.v_str_store.v_ml.chars, value->data.v_str_store.v_ml.size, value->pad};
  }
}

UnicodeHelper::self_view UnicodeHelper::AsView(const MATXScriptAny* value) {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value->code, TypeIndex::kRuntimeUnicode);
  return AsViewNoCheck(value);
}

MATXScriptAny UnicodeHelper::CopyFrom(const MATXScriptAny* value) {
  MATXScriptAny ret;
  auto view = UnicodeHelper::AsView(value);
  string_core<Unicode::value_type> str(view.data(), view.size(), view.category());
  str.MoveTo(&ret.data.v_str_store, &ret.pad);
  ret.code = TypeIndex::kRuntimeUnicode;
  return ret;
}

MATXScriptAny UnicodeHelper::CopyFrom(MATXScriptAny value) {
  return CopyFrom(&value);
}

void UnicodeHelper::Destroy(MATXScriptAny* value) noexcept {
  string_core<Unicode::value_type>::DestroyCHost(value);
}

}  // namespace runtime
}  // namespace matxscript
