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
#include <matxscript/runtime/regex/regex_ref.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>
#include <string>

#include <pcre.h>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/file_reader.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/regex/regex_private.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_OBJECT_TYPE(RegexNode);

RegexNode::RegexNode(const string_view& pattern,
                     bool ignore_case,
                     bool dotall,
                     bool extended,
                     bool anchored,
                     bool ucp)
    : pattern_(pattern.data(), pattern.size()),
      ignore_case_(ignore_case),
      dotall_(dotall),
      extended_(extended),
      anchored_(anchored),
      ucp_(ucp) {
  String errmsg;
  pcre_opt_ = 0;
  if (ignore_case) {
    pcre_opt_ |= PCRE_CASELESS;
  }
  if (dotall) {
    pcre_opt_ |= PCRE_DOTALL;
  }
  if (extended) {
    pcre_opt_ |= PCRE_EXTENDED;
  }
  if (anchored) {
    pcre_opt_ |= PCRE_ANCHORED;
  }
  auto pcre_opt = pcre_opt_;
  if (ucp) {
#ifdef PCRE_UCP
    pcre_opt |= PCRE_UCP;
#endif
  }

  re_ = regex::RegexPattern::Load(pattern_, &errmsg, pcre_opt);
}

List RegexNode::Split(const string_view& input) const {
  String errmsg;
  std::vector<String> rsl;
  List output;
  if (re_->Split(input, &rsl, &errmsg)) {
    output.reserve(rsl.size());
    auto mb = std::make_move_iterator(rsl.begin());
    auto me = std::make_move_iterator(rsl.end());
    for (; mb != me; ++mb) {
      output.push_back(std::move(*mb));
    }
  }
  return output;
}

List RegexNode::Split(const unicode_view& input) const {
  String errmsg;
  std::vector<String> rsl;
  List output;
  if (re_->Split(UnicodeHelper::Encode(input), &rsl, &errmsg)) {
    output.reserve(rsl.size());
    auto mb = std::make_move_iterator(rsl.begin());
    auto me = std::make_move_iterator(rsl.end());
    for (; mb != me; ++mb) {
      output.push_back(std::move(*mb).decode());
    }
  }
  return output;
}

List RegexNode::Split(const Any& input) const {
  switch (input.type_code()) {
    case TypeIndex::kRuntimeString: {
      return this->Split(input.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return this->Split(input.AsNoCheck<unicode_view>());
    } break;
    default: {
      THROW_PY_TypeError("Regex.split first arg must be str or bytes, not ", input.type_name());
      return List();
    } break;
  }
}

String RegexNode::Replace(const string_view& input, const string_view& repl) const {
  String errmsg;
  String result;
  if (re_->GSub(input, repl, &result, &errmsg)) {
    return result;
  } else {
    return input;
  }
}

Unicode RegexNode::Replace(const unicode_view& input, const unicode_view& repl) const {
  String errmsg;
  String result;
  if (re_->GSub(UnicodeHelper::Encode(input), UnicodeHelper::Encode(repl), &result, &errmsg)) {
    return result.decode();
  } else {
    return input;
  }
}

RTValue RegexNode::Replace(const Any& input, const Any& repl) const {
  switch (input.type_code()) {
    case TypeIndex::kRuntimeString: {
      return this->Replace(input.AsNoCheck<string_view>(), repl.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return this->Replace(input.AsNoCheck<unicode_view>(), repl.AsNoCheck<unicode_view>());
    } break;
    default: {
      THROW_PY_TypeError("Regex.replace first arg must be str or bytes, not ", input.type_name());
      return None;
    } break;
  }
}

Tuple RegexNode::Match(const string_view& input, int64_t offset) const {
  std::vector<String> match_array;
  std::unordered_map<String, int> match_named;
  String errmsg;
  bool is_matched = re_->Match(input, offset, &match_array, &match_named, &errmsg, pcre_opt_);

  List new_match_array;
  new_match_array.reserve(match_array.size());
  {
    auto mb = std::make_move_iterator(match_array.begin());
    auto me = std::make_move_iterator(match_array.end());
    for (; mb != me; ++mb) {
      new_match_array.push_back(std::move(*mb));
    }
  }

  Dict named_match_array;
  named_match_array.reserve(match_named.size());
  {
    auto mb = std::make_move_iterator(match_named.begin());
    auto me = std::make_move_iterator(match_named.end());
    for (; mb != me; ++mb) {
      named_match_array.emplace(mb->first, new_match_array[mb->second]);
    }
  }
  return Tuple::dynamic(new_match_array, named_match_array);
}

Tuple RegexNode::Match(const unicode_view& input, int64_t offset) const {
  std::vector<String> match_array;
  std::unordered_map<String, int> match_named;
  String errmsg;
  bool is_matched = re_->Match(UnicodeHelper::Encode(input.substr(offset)),
                               0,
                               &match_array,
                               &match_named,
                               &errmsg,
                               pcre_opt_);
  List new_match_array;
  new_match_array.reserve(match_array.size());
  {
    auto mb = std::make_move_iterator(match_array.begin());
    auto me = std::make_move_iterator(match_array.end());
    for (; mb != me; ++mb) {
      new_match_array.push_back(std::move(*mb).decode());
    }
  }

  Dict named_match_array;
  named_match_array.reserve(match_named.size());
  {
    auto mb = std::make_move_iterator(match_named.begin());
    auto me = std::make_move_iterator(match_named.end());
    for (; mb != me; ++mb) {
      named_match_array.emplace(mb->first.decode(), new_match_array[mb->second]);
    }
  }
  return Tuple::dynamic(new_match_array, named_match_array);
}

Tuple RegexNode::Match(const Any& input, int64_t offset) const {
  switch (input.type_code()) {
    case TypeIndex::kRuntimeString: {
      return this->Match(input.AsNoCheck<string_view>(), offset);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return this->Match(input.AsNoCheck<unicode_view>(), offset);
    } break;
    default: {
      THROW_PY_TypeError("Regex.match first arg must be str or bytes, not ", input.type_name());
      return Tuple{};
    } break;
  }
}

Regex::Regex(const string_view& pattern,
             bool ignore_case,
             bool dotall,
             bool extended,
             bool anchored,
             bool ucp) {
  data_ = make_object<RegexNode>(pattern, ignore_case, dotall, extended, anchored, ucp);
}

Regex::Regex(
    const Any& pattern, bool ignore_case, bool dotall, bool extended, bool anchored, bool ucp) {
  if (pattern.type_code() == TypeIndex::kRuntimeUnicode) {
    data_ = make_object<RegexNode>(UnicodeHelper::Encode(pattern.AsNoCheck<unicode_view>()),
                                   ignore_case,
                                   dotall,
                                   extended,
                                   anchored,
                                   ucp);
  } else {
    data_ = make_object<RegexNode>(
        pattern.As<string_view>(), ignore_case, dotall, extended, anchored, ucp);
  }
}

const RegexNode* Regex::operator->() const {
  return static_cast<const RegexNode*>(data_.get());
}

const RegexNode* Regex::get() const {
  return operator->();
}

List Regex::split(const string_view& input) const {
  MX_CHECK_DPTR(Regex);
  return d->Split(input);
}

List Regex::split(const unicode_view& input) const {
  MX_CHECK_DPTR(Regex);
  return d->Split(input);
}

List Regex::split(const Any& input) const {
  MX_CHECK_DPTR(Regex);
  return d->Split(input);
}

String Regex::replace(const string_view& input, const string_view& repl) const {
  MX_CHECK_DPTR(Regex);
  return d->Replace(input, repl);
}

Unicode Regex::replace(const unicode_view& input, const unicode_view& repl) const {
  MX_CHECK_DPTR(Regex);
  return d->Replace(input, repl);
}

RTValue Regex::replace(const Any& input, const Any& repl) const {
  MX_CHECK_DPTR(Regex);
  return d->Replace(input, repl);
}

Tuple Regex::match(const string_view& input, int64_t offset) const {
  MX_CHECK_DPTR(Regex);
  return d->Match(input, offset);
}

Tuple Regex::match(const unicode_view& input, int64_t offset) const {
  MX_CHECK_DPTR(Regex);
  return d->Match(input, offset);
}

Tuple Regex::match(const Any& input, int64_t offset) const {
  MX_CHECK_DPTR(Regex);
  return d->Match(input, offset);
}

template <>
bool IsConvertible<Regex>(const Object* node) {
  return node ? node->IsInstance<Regex::ContainerType>() : Regex::_type_is_nullable;
}

}  // namespace runtime
}  // namespace matxscript
