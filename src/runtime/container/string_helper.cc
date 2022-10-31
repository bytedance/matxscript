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
#include <matxscript/runtime/container/string_helper.h>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/ft_list.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

String StringHelper::Concat(self_view lhs, self_view rhs) {
  String ret;
  size_type lhs_size = lhs.size();
  size_type rhs_size = rhs.size();
  ret.reserve(lhs_size + rhs_size);
  ret.append(lhs.data(), lhs_size);
  ret.append(rhs.data(), rhs_size);
  return ret;
}

String StringHelper::Concat(std::initializer_list<self_view> args) {
  size_t cap = 0;
  auto itr = args.begin();
  auto itr_end = args.end();
  for (; itr != itr_end; ++itr) {
    cap += itr->size();
  }
  String ret;
  ret.resizeNoInit(cap);
  auto data = (String::pointer)ret.data();
  for (itr = args.begin(); itr != itr_end; ++itr) {
    String::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
  }
  return ret;
}

String StringHelper::Repeat(self_view sv, int64_t times) {
  times = MATXSCRIPT_UNLIKELY(times < 0) ? 0 : times;
  auto result_size = times * sv.length();
  String::ContainerType store(result_size, String::ContainerType::NoInit{});
  auto* data = (String::pointer)store.data();
  auto* src_data = sv.data();
  auto src_size = sv.size();
  for (int64_t i = 0; i < times; ++i) {
    String::traits_type::copy(data, src_data, src_size);
    data += src_size;
  }
  return String(std::move(store));
}

String StringHelper::Lower(self_view sv) {
  return AsciiDoLower(sv);
}

String StringHelper::Upper(self_view sv) {
  return AsciiDoUpper(sv);
}

bool StringHelper::Isdigit(self_view sv) noexcept {
  return AsciiIsDigit(sv);
}

bool StringHelper::Isalpha(self_view sv) noexcept {
  return AsciiIsAlpha(sv);
}

Unicode StringHelper::Decode(self_view sv) {
  return UTF8Decode(sv.begin(), sv.size());
}

bool StringHelper::Contains(self_view lhs, self_view rhs) noexcept {
  return lhs.find(rhs) != npos;
}

bool StringHelper::Contains(self_view sv, value_type c) noexcept {
  return sv.find(c) != npos;
}

int64_t StringHelper::GetItem(self_view sv, int64_t pos) {
  int64_t len = sv.size();
  MXCHECK((pos >= 0 && pos < len) || (pos < 0 && pos >= -len)) << "ValueError: index overflow";
  pos = slice_index_correction(pos, len);
  return int64_t{sv[pos]};
}

String StringHelper::GetSlice(self_view sv, int64_t b, int64_t e, int64_t step) {
  // TODO: change to noexcept
  MXCHECK_GT(step, 0) << "String.slice_load step must be gt 0";
  int64_t len = sv.size();
  b = slice_index_correction(b, len);
  e = slice_index_correction(e, len);
  if (e <= b) {
    return String();
  } else {
    if (step == 1) {
      return String(sv.begin() + b, sv.begin() + e);
    } else {
      String new_val;
      auto new_size = (e - b + step - 1) / step;
      new_val.resizeNoInit(new_size);
      auto* data = (String::pointer)new_val.data();
      auto itr_end = sv.begin() + e;
      for (auto itr = sv.begin() + b; itr < itr_end; itr += step) {
        *data++ = *itr;
      }
      return new_val;
    }
  }
}

List StringHelper::Split(self_view sv, self_view sep, int64_t maxsplit) {
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
    while (data_last < data_end && UCHAR_ISSPACE(*data_last)) {
      ++data_last;
    }
    auto* data_cur = data_last + 1;
    while (data_cur < data_end) {
      if (UCHAR_ISSPACE(*data_cur)) {
        if (maxsplit > 0) {
          ret_node->emplace_back(String(data_last, data_cur));
          data_last = data_cur + 1;
          // skip consecutive spaces
          while (data_last < data_end && UCHAR_ISSPACE(*data_last)) {
            ++data_last;
          }
          data_cur = data_last + 1;
          --maxsplit;
        } else {
          ret_node->emplace_back(String(data_last, data_end));
          data_last = data_end;
          break;
        }
      } else {
        ++data_cur;
      }
    }
    if (data_last < data_end) {
      ret_node->emplace_back(String(data_last, data_end));
    }
  } else {
    MXCHECK(!sep.empty()) << "ValueError: empty separator";
    size_type end;
    for (size_type start = 0; start < sv.size(); --maxsplit) {
      if (maxsplit > 0 && (end = sv.find(sep, start)) != npos) {
        ret_node->emplace_back(String(sv.substr(start, end - start)));
        start = end + sep.size();
      } else {
        ret_node->emplace_back(String(sv.substr(start)));
        break;
      }
    }
  }
  return ret;
}

String StringHelper::JoinStringList(self_view sv, std::initializer_list<String> il) {
  size_t cap = 0;
  auto itr = il.begin();
  auto itr_end = il.end();
  bool not_empty = itr != itr_end;
  if (not_empty) {
    cap += itr->size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    cap += sv.size();
    cap += itr->size();
  }
  itr = il.begin();
  String ret;
  ret.resizeNoInit(cap);
  String::pointer data = (String::pointer)ret.data();
  if (not_empty) {
    String::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    String::traits_type::copy(data, sv.data(), sv.size());
    data += sv.size();
    String::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
  }
  return ret;
}

String StringHelper::Join(self_view sv, const Any& iterable) {
  switch (iterable.type_code()) {
    case TypeIndex::kRuntimeIterator: {
      return Join(sv, iterable.AsObjectRefNoCheck<Iterator>());
    } break;
    case TypeIndex::kRuntimeList: {
      return Join(sv, iterable.AsObjectRefNoCheck<List>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      return Join(sv, iterable.AsObjectRefNoCheck<FTList<String>>());
    } break;
    default: {
      return Join(sv, Kernel_Iterable::make(iterable));
    } break;
  }
}

String StringHelper::Join(self_view sv, const Iterator& iter) {
  bool has_next = iter.HasNext();
  if (!has_next) {
    return String();
  }
  String ret(iter.Next(&has_next).As<string_view>());
  while (has_next) {
    ret.append(sv);
    ret.append(iter.Next(&has_next).As<string_view>());
  }
  return ret;
}

String StringHelper::Join(self_view sv, const List& list) {
  size_t cap = 0;
  auto itr = list.begin();
  auto itr_end = list.end();
  bool not_empty = itr != itr_end;
  if (not_empty) {
    cap += itr->As<string_view>().size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    cap += sv.size();
    cap += itr->As<string_view>().size();
  }
  itr = list.begin();
  String ret;
  ret.resizeNoInit(cap);
  auto data = (String::pointer)ret.data();
  if (not_empty) {
    const auto& view = itr->AsNoCheck<string_view>();
    String::traits_type::copy(data, view.data(), view.size());
    data += view.size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    String::traits_type::copy(data, sv.data(), sv.size());
    data += sv.size();
    const auto& view = itr->AsNoCheck<string_view>();
    String::traits_type::copy(data, view.data(), view.size());
    data += view.size();
  }
  return ret;
}

String StringHelper::Join(self_view sv, const FTList<String>& list) {
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
  String ret;
  ret.resizeNoInit(cap);
  auto data = (String::pointer)ret.data();
  if (not_empty) {
    String::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
    ++itr;
  }
  for (; itr != itr_end; ++itr) {
    String::traits_type::copy(data, sv.data(), sv.size());
    data += sv.size();
    String::traits_type::copy(data, itr->data(), itr->size());
    data += itr->size();
  }
  return ret;
}

String StringHelper::Replace(self_view sv, self_view old_s, self_view new_s, int64_t count) {
  if (count < 0) {
    count = std::numeric_limits<int64_t>::max();
  }
  String ret;
  // What's the proper value of the capacity?
  if (new_s.length() > old_s.length()) {
    ret.reserve(sv.size() * 1.5);
  } else {
    ret.reserve(sv.size());
  }
  size_type current = 0, next;
  while ((next = sv.find(old_s, current)) != npos && count > 0) {
    ret.append(sv.data() + current, next - current);
    ret.append(new_s.data(), new_s.size());
    current = next + old_s.size();
    --count;
  }
  ret.append(sv.data() + current, sv.size() - current);
  return ret;
}

bool StringHelper::EndsWith(self_view sv, self_view suffix, int64_t start, int64_t end) noexcept {
  // [start:end)
  int64_t suffix_len = suffix.size();
  int64_t len = sv.size();
  start = slice_index_correction(start, len);
  end = slice_index_correction(end, len);

  if (end - start < suffix_len || start >= len) {
    return false;
  }
  return suffix.compare(sv.SubStrNoCheck(end - suffix_len, suffix_len)) == 0;
}

bool StringHelper::EndsWith(self_view sv, const Tuple& suffixes, int64_t start, int64_t end) {
  for (const auto& suffix : suffixes) {
    if (!suffix.Is<string_view>()) {
      THROW_PY_TypeError("a bytes-like object is required, not '", suffix.type_name(), "'");
    }
    if (EndsWith(sv, suffix.AsNoCheck<string_view>(), start, end)) {
      return true;
    }
  }
  return false;
}

bool StringHelper::EndsWith(self_view sv,
                            const Any& suffix_or_suffixes,
                            int64_t start,
                            int64_t end) {
  switch (suffix_or_suffixes.type_code()) {
    case TypeIndex::kRuntimeString: {
      return EndsWith(sv, suffix_or_suffixes.AsNoCheck<string_view>(), start, end);
    } break;
    case TypeIndex::kRuntimeTuple: {
      return EndsWith(sv, suffix_or_suffixes.AsObjectViewNoCheck<Tuple>().data(), start, end);
    } break;
    default: {
      THROW_PY_TypeError("endswith first arg must be bytes or a tuple of bytes, not ",
                         suffix_or_suffixes.type_name());
      return false;
    } break;
  }
}

bool StringHelper::StartsWith(self_view sv, self_view prefix, int64_t start, int64_t end) noexcept {
  // [start:end)
  int64_t prefix_len = prefix.size();
  int64_t len = sv.size();
  start = slice_index_correction(start, len);
  end = slice_index_correction(end, len);
  if (end - start < prefix_len || start >= len) {
    return false;
  }
  return prefix.compare(sv.SubStrNoCheck(start, prefix_len)) == 0;
}

bool StringHelper::StartsWith(self_view sv, const Tuple& prefixes, int64_t start, int64_t end) {
  for (const auto& prefix : prefixes) {
    if (!prefix.Is<string_view>()) {
      THROW_PY_TypeError("a bytes-like object is required, not '", prefix.type_name(), "'");
    }
    MXCHECK(prefix.IsString()) << "elements of `suffixes` should be String.";
    if (StartsWith(sv, prefix.As<string_view>(), start, end)) {
      return true;
    }
  }
  return false;
}

bool StringHelper::StartsWith(self_view sv,
                              const Any& prefix_or_prefixes,
                              int64_t start,
                              int64_t end) {
  switch (prefix_or_prefixes.type_code()) {
    case TypeIndex::kRuntimeString: {
      return StartsWith(sv, prefix_or_prefixes.AsNoCheck<string_view>(), start, end);
    } break;
    case TypeIndex::kRuntimeTuple: {
      return StartsWith(sv, prefix_or_prefixes.AsObjectViewNoCheck<Tuple>().data(), start, end);
    } break;
    default: {
      THROW_PY_TypeError("startswith first arg must be bytes or a tuple of bytes, not",
                         prefix_or_prefixes.type_name());
      return false;
    } break;
  }
}

String StringHelper::LStrip(self_view sv, self_view chars) {
  auto* data = sv.data();
  auto* data_end = sv.data() + sv.size();
  if (chars.data() == nullptr) {
    while (data < data_end && UCHAR_ISSPACE(*data)) {
      ++data;
    }
    if (data < data_end) {
      return String(data, data_end - data);
    } else {
      return String{};
    }
  } else {
    while (data < data_end && chars.find(*data) != chars.npos) {
      ++data;
    }
    if (data < data_end) {
      return String(data, data_end - data);
    } else {
      return String{};
    }
  }
}

String StringHelper::RStrip(self_view sv, self_view chars) {
  auto* data_begin = sv.data();
  auto* data = data_begin + sv.size() - 1;
  if (chars.data() == nullptr) {
    while (data >= data_begin && UCHAR_ISSPACE(*data)) {
      --data;
    }
    if (data >= data_begin) {
      return String(data_begin, data - data_begin + 1);
    } else {
      return String{};
    }
  } else {
    while (data >= data_begin && chars.find(*data) != chars.npos) {
      --data;
    }
    if (data >= data_begin) {
      return String(data_begin, data - data_begin + 1);
    } else {
      return String{};
    }
  }
}

String StringHelper::Strip(self_view sv, self_view chars) {
  auto* data_begin = sv.data();
  auto* data_end = sv.data() + sv.size();
  auto* data_left = data_begin;
  auto* data_right = data_begin + sv.size() - 1;
  if (chars.data() == nullptr) {
    while (data_left < data_end && UCHAR_ISSPACE(*data_left)) {
      ++data_left;
    }
    while (data_right > data_left && UCHAR_ISSPACE(*data_right)) {
      --data_right;
    }
    if (data_right >= data_left) {
      return String(data_left, data_right - data_left + 1);
    } else {
      return String{};
    }
  } else {
    while (data_left < data_end && chars.find(*data_left) != chars.npos) {
      ++data_left;
    }
    while (data_right > data_left && chars.find(*data_right) != chars.npos) {
      --data_right;
    }
    if (data_right >= data_left) {
      return String(data_left, data_right - data_left + 1);
    } else {
      return String{};
    }
  }
}

int64_t StringHelper::Count(self_view sv, self_view x, int64_t start, int64_t end) noexcept {
  int64_t x_len = x.size();
  int64_t len = sv.size();
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

StringHelper::self_view StringHelper::AsViewNoCheck(const MATXScriptAny* value) noexcept {
  if (value->pad >= 0) {
    return string_view{(char*)value->data.v_str_store.v_small_bytes, size_t(value->pad), 0};
  } else {
    return string_view{
        (char*)value->data.v_str_store.v_ml.bytes, value->data.v_str_store.v_ml.size, value->pad};
  }
}

StringHelper::self_view StringHelper::AsView(const MATXScriptAny* value) {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value->code, TypeIndex::kRuntimeString);
  return AsViewNoCheck(value);
}

MATXScriptAny StringHelper::CopyFrom(const MATXScriptAny* value) {
  MATXScriptAny ret;
  auto view = StringHelper::AsView(value);
  string_core<String::value_type> str(view.data(), view.size(), view.category());
  str.MoveTo(&ret.data.v_str_store, &ret.pad);
  ret.code = TypeIndex::kRuntimeString;
  return ret;
}

MATXScriptAny StringHelper::CopyFrom(MATXScriptAny value) {
  return CopyFrom(&value);
}

void StringHelper::Destroy(MATXScriptAny* value) noexcept {
  string_core<String::value_type>::DestroyCHost(value);
}

}  // namespace runtime
}  // namespace matxscript
