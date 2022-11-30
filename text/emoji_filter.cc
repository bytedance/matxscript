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
#include "emoji_filter.h"

#include "common_funcs.h"

#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/type_helper_macros.h>

namespace matxscript {
namespace runtime {
namespace extension {
namespace emoji {

// declare resource
extern const char* const UNICODE_EMOJI_EN[][2];
extern const char* const UNICODE_EMOJI_ALIAS_EN[][2];

template <typename DictContainerType>
static void AppendResource(const char* const resource[], DictContainerType& dic) {
  uint32_t len = 0;
  while (resource[len] != nullptr) {
    dic[String(resource[len])] = 0;
    ++len;
  }
}

template <typename DictContainerType>
static void AppendResource(const char* const resource[][2], int idx, DictContainerType& dic) {
  uint32_t len = 0;
  while (resource[len][idx] != nullptr) {
    dic[String(resource[len][idx])] = 0;
    ++len;
  }
}

template <typename IteratorType, typename DictContainerType>
static void AppendResource(IteratorType first, IteratorType last, DictContainerType& dic) {
  while (first != last) {
    dic[commons::details::GetString(*first, __FILE__, __LINE__)] = 0;
    ++first;
  }
}

// just for static link
EmojiFilter::EmojiFilter(Options opt) {
  opt_ = std::move(opt);
  std::map<String, int> dic;
  if (opt_.unicode) {
    AppendResource(UNICODE_EMOJI_EN, 1, dic);
  }
  if (opt_.unicode_trans) {
    AppendResource(UNICODE_EMOJI_EN, 0, dic);
  }
  if (opt_.unicode_trans_alias) {
    AppendResource(UNICODE_EMOJI_ALIAS_EN, 0, dic);
  }
  switch (opt_.user_codes.type_code()) {
    case TypeIndex::kRuntimeNullptr: {
    } break;
    case TypeIndex::kRuntimeList: {
      auto li = opt_.user_codes.AsNoCheck<List>();
      AppendResource(li.begin(), li.end(), dic);
    } break;
    case TypeIndex::kRuntimeTuple: {
      auto tup = opt_.user_codes.AsNoCheck<Tuple>();
      AppendResource(tup.begin(), tup.end(), dic);
    } break;
    case TypeIndex::kRuntimeSet: {
      auto s = opt_.user_codes.AsNoCheck<Set>();
      AppendResource(s.begin(), s.end(), dic);
    } break;
    case TypeIndex::kRuntimeDict: {
      auto d = opt_.user_codes.AsNoCheck<Dict>();
      AppendResource(d.key_begin(), d.key_end(), dic);
    } break;
    default: {
      THROW_PY_TypeError("expect user_emojis is 'list' 'tuple' or 'set' type, but get '",
                         opt_.user_codes.type_name(),
                         "'");
    } break;
  }
  emoji_codes_ = std::make_shared<PrefixMapping>(dic);
}

String EmojiFilter::Replace(const string_view& str, const string_view& repl, bool keep_all) const {
  if (repl.empty()) {
    return Filter(str);
  } else {
    String result;
    auto ptr = str.data();
    int64_t len = str.size();
    result.reserve(len);
    bool last_is_emoji = false;
    while (len > 0) {
      auto match_len = emoji_codes_->PrefixSearch(ptr, len, nullptr);
      if (match_len <= 0) {
        result.push_back(ptr[0]);
        ++ptr;
        --len;
        last_is_emoji = false;
      } else {
        if (keep_all || !last_is_emoji) {
          result.append(repl);
        }
        ptr += match_len;
        len -= match_len;
        last_is_emoji = true;
      }
    }
    return result;
  }
}

String EmojiFilter::Filter(const string_view& str) const {
  String result;
  auto ptr = str.data();
  int64_t len = str.size();
  result.reserve(len);

  while (len > 0) {
    auto match_len = emoji_codes_->PrefixSearch(ptr, len, nullptr);
    if (match_len <= 0) {
      result.push_back(ptr[0]);
      ++ptr;
      --len;
    } else {
      ptr += match_len;
      len -= match_len;
    }
  }

  return result;
}

using text_emoji_EmojiFilter = EmojiFilter;

MATX_REGISTER_NATIVE_OBJECT(text_emoji_EmojiFilter)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      EmojiFilter::Options options;
      options.unicode = MATXSCRIPT_TYPE_AS(args[0], bool);
      options.unicode_trans = MATXSCRIPT_TYPE_AS(args[1], bool);
      options.unicode_trans_alias = MATXSCRIPT_TYPE_AS(args[2], bool);
      if (args.size() == 4) {
        options.user_codes = MATXSCRIPT_TYPE_AS(args[3], RTValue);
      }
      return std::make_shared<EmojiFilter>(std::move(options));
    })
    .def("filter",
         [](void* self, PyArgs args) -> RTValue {
           switch (args[0].type_code()) {
             case TypeIndex::kRuntimeString: {
               return reinterpret_cast<EmojiFilter*>(self)->Filter(
                   args[0].AsNoCheck<string_view>());
             } break;
             case TypeIndex::kRuntimeUnicode: {
               auto s = UTF8Encode(args[0].AsNoCheck<unicode_view>());
               return UTF8Decode(reinterpret_cast<EmojiFilter*>(self)->Filter(s));
             } break;
             default: {
               auto ty_name = args[0].type_name();
               std::string errmsg;
               errmsg.append("emoji.filter(): expect type is 'py::str' or 'py::bytes', but get '");
               errmsg.append(ty_name.data(), ty_name.size());
               errmsg.append("'");
               throw TypeError(__FILE__, __LINE__, std::move(errmsg));
             }
           }
         })
    .def("replace",
         [](void* self, PyArgs args) -> RTValue {
           bool keep_all = MATXSCRIPT_TYPE_AS(args[2], bool);
           switch (args[0].type_code()) {
             case TypeIndex::kRuntimeString: {
               auto repl = MATXSCRIPT_TYPE_AS(args[1], string_view);
               return reinterpret_cast<EmojiFilter*>(self)->Replace(
                   args[0].AsNoCheck<string_view>(), repl, keep_all);
             } break;
             case TypeIndex::kRuntimeUnicode: {
               auto s = UTF8Encode(args[0].AsNoCheck<unicode_view>());
               auto repl = UTF8Encode(MATXSCRIPT_TYPE_AS(args[1], unicode_view));
               return UTF8Decode(reinterpret_cast<EmojiFilter*>(self)->Replace(s, repl, keep_all));
             } break;
             default: {
               auto ty_name = args[0].type_name();
               std::string errmsg;
               errmsg.append("emoji.replace(): expect type is 'py::str' or 'py::bytes', but get '");
               errmsg.append(ty_name.data(), ty_name.size());
               errmsg.append("'");
               throw TypeError(__FILE__, __LINE__, std::move(errmsg));
             }
           }
         })
    .def("check_pos", [](void* self, PyArgs args) -> RTValue {
      int64_t pos = MATXSCRIPT_TYPE_AS(args[1], int64_t);
      switch (args[0].type_code()) {
        case TypeIndex::kRuntimeString: {
          auto s = args[0].AsNoCheck<string_view>();
          pos = slice_index_correction(pos, s.size());
          return reinterpret_cast<EmojiFilter*>(self)->CheckPos(s.data(), s.size(), pos);
        } break;
        case TypeIndex::kRuntimeUnicode: {
          auto us = args[0].AsNoCheck<unicode_view>();
          pos = slice_index_correction(pos, us.size());
          auto s = UTF8Encode(us.data() + pos, us.size());
          auto len = reinterpret_cast<EmojiFilter*>(self)->CheckPos(s.data(), s.size(), 0);
          return int64_t(UTF8CharCounts(string_view(s.data(), len)));
        } break;
        default: {
          auto ty_name = args[0].type_name();
          std::string errmsg;
          errmsg.append("emoji.check_begin(): expect type is 'py::str' or 'py::bytes', but get '");
          errmsg.append(ty_name.data(), ty_name.size());
          errmsg.append("'");
          throw TypeError(__FILE__, __LINE__, std::move(errmsg));
        }
      }
    });

}  // namespace emoji
}  // namespace extension
}  // namespace runtime
}  // namespace matxscript
