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
#include <matxscript/runtime/container/string.h>

#include <initializer_list>

#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/ft_list.h>
#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

bool String::isSane() const noexcept {
  return begin() <= end() && empty() == (size() == 0) && empty() == (begin() == end()) &&
         size() <= max_size() && capacity() <= max_size() && size() <= capacity() &&
         begin()[size()] == '\0';
}

#ifdef MATXSCRIPT_RUNTIME_STRING_UNICODE_ENABLE_INVARIANT_CHECK
namespace {
struct Invariant {
  Invariant& operator=(const Invariant&) = delete;
  explicit Invariant(const String& s) noexcept : s_(s) {
    assert(s_.isSane());
  }
  ~Invariant() noexcept {
    assert(s_.isSane());
  }

 private:
  const String& s_;
};
}  // namespace

#define STRING_INVARIANT_CHECK(s) Invariant invariant_checker(s)
#else
#define STRING_INVARIANT_CHECK(s)
#endif  // MATXSCRIPT_RUNTIME_STRING_UNICODE_ENABLE_INVARIANT_CHECK

/******************************************************************************
 * Generic String Iterator
 *****************************************************************************/

class StringIteratorNode : public IteratorNode {
 public:
  explicit StringIteratorNode(String container) noexcept
      : container_(std::move((container))), first_(container_.cbegin()), last_(container_.cend()) {
  }
  ~StringIteratorNode() noexcept = default;

  bool HasNext() const override {
    return first_ != last_;
  }
  RTValue Next() override {
    return int64_t(*(first_++));
  }
  RTValue Next(bool* has_next) override {
    int64_t ret = (unsigned char)(*(first_++));
    *has_next = (first_ != last_);
    return ret;
  }
  RTView NextView(bool* has_next, RTValue* holder_or_null) override {
    int64_t ret = *(first_++);
    *has_next = (first_ != last_);
    return ret;
  }

  int64_t Distance() const override {
    return std::distance(first_, last_);
  }

  uint64_t HashCode() const override {
    return std::hash<String>()(container_);
  }

 public:
  String container_;
  String::const_iterator first_;
  String::const_iterator last_;
};

Iterator String::iter() const {
  auto data = make_object<StringIteratorNode>(*this);
  return Iterator(std::move(data));
}

/******************************************************************************
 * String container
 *****************************************************************************/

const typename String::size_type String::npos;

const String::value_type* String::data() const noexcept {
  return data_.data();
}

String::operator String::self_view() const noexcept {
  return self_view{data_.data(), data_.size(), data_.category()};
}

String::self_view String::view() const noexcept {
  return self_view{data_.data(), data_.size(), data_.category()};
}

// MATXScriptAny
void String::MoveTo(MATXScriptAny* value) noexcept {
  data_.MoveTo(&value->data.v_str_store, &value->pad);
  value->code = TypeIndex::kRuntimeString;
}

String String::MoveFromNoCheck(MATXScriptAny* value) noexcept {
  return String(ContainerType::MoveFromCHost(value));
}

String& String::operator=(const String& other) {
  STRING_INVARIANT_CHECK(*this);
  if (MATXSCRIPT_UNLIKELY(&other == this)) {
    return *this;
  }
  if (other.data_.category() == ContainerType::Category::isLarge) {
    ContainerType(other.data_).swap(data_);
    return *this;
  }
  return assign(other.data_.data(), other.data_.size());
}

String& String::operator=(String&& other) noexcept {
  if (MATXSCRIPT_UNLIKELY(&other == this)) {
    // Compatibility with std::basic_string<>,
    // C++11 21.4.2 [string.cons] / 23 requires self-move-assignment support.
    return *this;
  }
  // No need of this anymore
  this->~String();
  // Move the goner into this
  new (&data_) ContainerType(std::move(other.data_));
  return *this;
}

// assign from other
String& String::operator=(const std::string& other) {
  return assign(other.data(), other.size());
}

String& String::operator=(self_view other) {
  if (other.category() == ContainerType::Category::isLarge) {
    ContainerType(other.data(), other.size(), other.category()).swap(data_);
    return *this;
  }
  return assign(other.data(), other.size());
}

String& String::operator=(const value_type* other) {
  return other ? assign(other, std::char_traits<value_type>::length(other)) : *this;
}

String& String::operator=(value_type other) {
  ContainerType(&other, 1).swap(data_);
  return *this;
}

String& String::operator=(std::initializer_list<value_type> il) {
  return assign(il.begin(), il.size());
}

int String::compare(const String& other) const noexcept {
  return view().compare(other.view());
}

int String::compare(const std::string& other) const noexcept {
  return view().compare(other);
}

int String::compare(const char* other) const noexcept {
  return view().compare(other);
}

const char* String::c_str() const noexcept {
  return data_.c_str();
}

int64_t String::size() const noexcept {
  return data_.size();
}

int64_t String::length() const noexcept {
  return size();
}

bool String::empty() const noexcept {
  return size() == 0;
}

String String::repeat(int64_t times) const {
  return StringHelper::Repeat(view(), times);
}

String String::lower() const {
  return StringHelper::Lower(view());
}

String String::upper() const {
  return StringHelper::Upper(view());
}

bool String::isdigit() const noexcept {
  return AsciiIsDigit(view());
}

bool String::isalpha() const noexcept {
  return AsciiIsAlpha(view());
}

const char& String::front() const noexcept {
  return *(data_.data());
}

const char& String::back() const noexcept {
  return *(data_.data() + data_.size() - 1);
}

char& String::front() {
  return *(data_.mutableData());
}

char& String::back() {
  assert(!empty());
  return *(data_.mutableData() + data_.size() - 1);
}

void String::pop_back() {
  STRING_INVARIANT_CHECK(*this);
  assert(!empty());
  data_.shrink(1);
}

void String::resizeNoInit(size_type n) {
  STRING_INVARIANT_CHECK(*this);
  size_type len = data_.size();
  if (n <= len) {
    data_.shrink(len - n);
  } else {
    auto const delta = n - len;
    data_.expandNoinit(delta);
  }
}

void String::resize(size_type n, char c) {
  STRING_INVARIANT_CHECK(*this);
  size_type len = data_.size();
  if (n <= len) {
    data_.shrink(len - n);
  } else {
    auto const delta = n - len;
    value_type* p = data_.expandNoinit(delta);
    string_detail::podFill(p, p + delta, c);
  }
}

int64_t String::capacity() const noexcept {
  return data_.capacity();
}

void String::reserve(size_type res_arg) {
  data_.reserve(res_arg);
}

void String::shrink_to_fit() {
  if (data_.capacity() < data_.size() * 3 / 2) {
    return;
  }
  ContainerType(data_.data(), data_.size()).swap(data_);
}

void String::clear() {
  resize(0);
}

Unicode String::decode() const {
  self_view str = view();
  return UTF8Decode(str.data(), str.size());
}

const char& String::operator[](size_type pos) const noexcept {
  return *(data_.data() + pos);
}

int64_t String::get_item(int64_t pos) const {
  return StringHelper::GetItem(view(), pos);
}

String String::get_slice(int64_t b, int64_t e, int64_t step) const {
  return StringHelper::GetSlice(view(), b, e, step);
}

char& String::operator[](size_type pos) {
  return *(data_.mutableData() + pos);
}

const char& String::at(size_type n) const {
  if (n >= size()) {
    throw std::out_of_range("String: index out of range");
  }
  return *(data_.data() + n);
}

char& String::at(size_type n) {
  if (n >= size()) {
    throw std::out_of_range("String: index out of range");
  }
  return *(data_.mutableData() + n);
}

String& String::operator+=(const String& str) {
  return append(str);
}

String& String::operator+=(const char* s) {
  return append(s);
}

String& String::operator+=(const char c) {
  push_back(c);
  return *this;
}

String& String::operator+=(std::initializer_list<char> il) {
  return append(self_view(il.begin(), il.size()));
}

String& String::append(self_view str) {
  return append(str.data(), str.size());
}

String& String::append(self_view str, size_type pos, size_type n) {
  return append(str.substr(pos, n));
}

String& String::append(const char* s, size_type n) {
  STRING_INVARIANT_CHECK(*this);
  if (MATXSCRIPT_UNLIKELY(!n)) {
    // Unlikely but must be done
    return *this;
  }
  auto const oldSize = size();
  auto const oldData = data();
  auto pData = data_.expandNoinit(n, /* expGrowth = */ true);

  // Check for aliasing (rare). We could use "<=" here but in theory
  // those do not work for pointers unless the pointers point to
  // elements in the same array. For that reason we use
  // std::less_equal, which is guaranteed to offer a total order
  // over pointers. See discussion at http://goo.gl/Cy2ya for more
  // info.
  std::less_equal<const value_type*> le;
  if (MATXSCRIPT_UNLIKELY(le(oldData, s) && !le(oldData + oldSize, s))) {
    assert(le(s + n, oldData + oldSize));
    // expandNoinit() could have moved the storage, restore the source.
    s = data() + (s - oldData);
    string_detail::podMove(s, s + n, pData);
  } else {
    string_detail::podCopy(s, s + n, pData);
  }
  assert(size() == oldSize + n);
  return *this;
}

String& String::append(const char* s) {
  return s ? append(s, std::char_traits<char>::length(s)) : *this;
}

String& String::append(size_type n, char c) {
  STRING_INVARIANT_CHECK(*this);
  auto pData = data_.expandNoinit(n, /* expGrowth = */ true);
  string_detail::podFill(pData, pData + n, c);
  return *this;
}

void String::push_back(const char c) {
  STRING_INVARIANT_CHECK(*this);
  data_.push_back(c);
}

String& String::assign(const String& str) {
  if (MATXSCRIPT_UNLIKELY(&str == this)) {
    return *this;
  }
  if (str.data_.category() == ContainerType::Category::isLarge) {
    ContainerType(str.data_).swap(data_);
    return *this;
  }
  return assign(str.data_.data(), str.data_.size());
}

String& String::assign(String&& str) noexcept {
  return *this = std::move(str);
}

String& String::assign(const String& str, size_type pos, size_type n) {
  auto sub_view = str.view().substr(pos, n);
  assign(sub_view.data(), sub_view.size());
  return *this;
}

String& String::assign(const value_type* s, size_type n) {
  STRING_INVARIANT_CHECK(*this);

  auto self_len = data_.size();
  if (n == 0) {
    data_.shrink(self_len);
  } else if (self_len >= n) {
    // s can alias this, we need to use podMove.
    string_detail::podMove(s, s + n, data_.mutableData());
    data_.shrink(self_len - n);
    assert(size() == n);
  } else {
    // If n is larger than size(), s cannot alias this string's
    // storage.
    if (data_.isShared()) {
      ContainerType(s, n).swap(data_);
    } else {
      // Do not use exponential growth here: assign() should be tight,
      // to mirror the behavior of the equivalent constructor.
      data_.expandNoinit(n - self_len);
      string_detail::podCopy(s, s + n, data_.data());
    }
  }

  assert(size() == n);
  return *this;
}

String& String::assign(const value_type* s) {
  return s ? assign(s, std::char_traits<value_type>::length(s)) : *this;
}

String& String::assign(const string_view& s) {
  if (s.category() == ContainerType::Category::isLarge) {
    ContainerType(s.data(), s.size(), s.category()).swap(data_);
    return *this;
  }
  return assign(s.data(), s.size());
}

String& String::insert(size_type pos1, const String& str) {
  return insert(pos1, str.view());
}

String& String::insert(size_type pos1, const String& str, size_type pos2, size_type n) {
  return insert(pos1, str.view().substr(pos2, n));
}

String& String::insert(size_type pos, self_view s) {
  STRING_INVARIANT_CHECK(*this);
  auto n = s.size();
  auto oldSize = size();
  data_.expandNoinit(n, /* expGrowth = */ true);
  auto b = begin();
  string_detail::podMove(b + pos, b + oldSize, b + pos + n);
  std::copy(s.begin(), s.end(), b + pos);
  return *this;
}

String& String::insert(size_type pos, size_type n, char c) {
  STRING_INVARIANT_CHECK(*this);
  auto oldSize = size();
  data_.expandNoinit(n, /* expGrowth = */ true);
  auto b = begin();
  string_detail::podMove(b + pos, b + oldSize, b + pos + n);
  string_detail::podFill(b + pos, b + pos + n, c);
  return *this;
}

String& String::insert(size_type pos, const value_type* str) {
  return insert(pos, self_view(str));
}

String& String::insert(size_type pos, const value_type* str, size_type n) {
  return insert(pos, self_view(str, n));
}

List String::split(self_view sep, int64_t maxsplit) const {
  return StringHelper::Split(view(), sep, maxsplit);
}

String String::join(const RTValue& iterable) const {
  return StringHelper::Join(view(), iterable);
}

String String::join(const Iterator& iter) const {
  return StringHelper::Join(view(), iter);
}

String String::join(const List& list) const {
  return StringHelper::Join(view(), list);
}

String String::join(const FTList<String>& list) const {
  return StringHelper::Join(view(), list);
}

String::operator std::string() const {
  return view().operator std::string();
}

String& String::erase(size_type pos, size_type n) {
  STRING_INVARIANT_CHECK(*this);
  if (pos > size()) {
    throw std::out_of_range("String: index out of range");
  }
  n = std::min(n, size_type(size()) - pos);
  std::copy(begin() + pos + n, end(), begin() + pos);
  resize(size() - n);
  return *this;
}

String String::replace(self_view old_s, self_view new_s, int64_t count) const {
  return StringHelper::Replace(view(), old_s, new_s, count);
}

String& String::replace(size_type pos1, size_type n1, self_view s, size_type pos2, size_type n2) {
  return replace(pos1, n1, s.substr(pos2, n2));
}

String& String::replace(size_type pos, size_type n1, self_view s) {
  STRING_INVARIANT_CHECK(*this);
  size_type len = size();
  if (pos > len) {
    throw std::out_of_range("String: index out of range");
  }
  n1 = std::min(n1, len - pos);
  String tmp;
  tmp.reserve(len - n1 + s.size());
  tmp.append(data(), pos).append(s.data(), s.size()).append(data() + pos + n1, len - pos - n1);
  *this = std::move(tmp);
  return *this;
}

String& String::replace(size_type pos, size_type n1, self_view s, size_type n2) {
  return replace(pos, n1, s.substr(0, n2));
}

String& String::replace(size_type pos, size_type n1, size_type n2, value_type c) {
  STRING_INVARIANT_CHECK(*this);
  size_type len = size();
  if (pos > len) {
    throw std::out_of_range("String: index out of range");
  }
  n1 = std::min(n1, len - pos);
  String tmp;
  tmp.reserve(len - n1 + n2);
  tmp.append(data(), pos).append(n2, c).append(data() + pos + n1, len - pos - n1);
  *this = std::move(tmp);
  return *this;
}

bool String::contains(self_view str) const noexcept {
  return StringHelper::Contains(view(), str);
}

String::size_type String::find(self_view str, size_type pos) const {
  return view().find(str, pos);
}

String::size_type String::find(value_type c, size_type pos) const {
  return view().find(c, pos);
}

String::size_type String::rfind(self_view str, size_type pos) const {
  return view().rfind(str, pos);
}

String::size_type String::rfind(value_type c, size_type pos) const {
  return view().rfind(c, pos);
}

String::size_type String::find_first_of(self_view str, size_type pos) const {
  return view().find_first_of(str, pos);
}

String::size_type String::find_first_of(value_type c, size_type pos) const {
  return view().find_first_of(c, pos);
}

String::size_type String::find_last_of(self_view str, size_type pos) const {
  return view().find_last_of(str, pos);
}

String::size_type String::find_last_of(value_type c, size_type pos) const {
  return view().find_last_of(c, pos);
}

String::size_type String::find_first_not_of(self_view str, size_type pos) const {
  return view().find_first_not_of(str, pos);
}

String::size_type String::find_first_not_of(value_type c, size_type pos) const {
  return view().find_first_not_of(c, pos);
}

String::size_type String::find_last_not_of(self_view str, size_type pos) const {
  return view().find_last_not_of(str, pos);
}

String::size_type String::find_last_not_of(value_type c, size_type pos) const {
  return view().find_last_not_of(c, pos);
}

String String::substr(size_type pos, size_type n) const {
  return String(view().substr(pos, n));
}

bool String::endswith(self_view suffix, int64_t start, int64_t end) const noexcept {
  return StringHelper::EndsWith(view(), suffix, start, end);
}

bool String::endswith(const Tuple& suffixes, int64_t start, int64_t end) const {
  return StringHelper::EndsWith(view(), suffixes, start, end);
}

bool String::endswith(const Any& suffix_or_suffixes, int64_t start, int64_t end) const {
  return StringHelper::EndsWith(view(), suffix_or_suffixes, start, end);
}

bool String::startswith(self_view prefix, int64_t start, int64_t end) const noexcept {
  return StringHelper::StartsWith(view(), prefix, start, end);
}

bool String::startswith(const Tuple& prefixes, int64_t start, int64_t end) const {
  return StringHelper::StartsWith(view(), prefixes, start, end);
}

bool String::startswith(const Any& prefix_or_prefixes, int64_t start, int64_t end) const {
  return StringHelper::StartsWith(view(), prefix_or_prefixes, start, end);
}

String String::lstrip(self_view chars) const {
  return StringHelper::LStrip(view(), chars);
}

String String::rstrip(self_view chars) const {
  return StringHelper::RStrip(view(), chars);
}

String String::strip(self_view chars) const {
  return StringHelper::Strip(view(), chars);
}

int64_t String::count(self_view x, int64_t start, int64_t end) const noexcept {
  return StringHelper::Count(view(), x, start, end);
}

String String::Concat(self_view lhs, self_view rhs) {
  return StringHelper::Concat(lhs, rhs);
}

String String::Concat(std::initializer_list<self_view> args) {
  return StringHelper::Concat(args);
}

/*
template <>
bool IsConvertible<String>(const Object* node) {
  return node ? node->IsInstance<String::ContainerType>() : String::_type_is_nullable;
}

MATX_REGISTER_GLOBAL("runtime.String").set_body_typed([](std::string str) {
  return String(std::move(str));
});

MATX_REGISTER_GLOBAL("runtime.GetFFIString").set_body_typed([](String str) {
  return std::string(str);
});

// runtime member function
MATX_REGISTER_GLOBAL("runtime.StringLen").set_body_typed([](String str) {
  return static_cast<int64_t>(str.size());
});

MATX_REGISTER_GLOBAL("runtime.StringAdd").set_body_typed([](String lhs, String rhs) {
  return lhs + rhs;
});

MATX_REGISTER_GLOBAL("runtime.StringEqual").set_body_typed([](String lhs, String rhs) {
  return lhs == rhs;
});

MATX_REGISTER_GLOBAL("runtime.StringHash").set_body_typed([](String str) {
  return static_cast<int64_t>(std::hash<String>()(str));
});
*/

/******************************************************************************
 * String iterators
 *****************************************************************************/

typename String::iterator String::begin() {
  return data_.mutableData();
}

typename String::const_iterator String::begin() const noexcept {
  return data_.data();
}

typename String::const_iterator String::cbegin() const noexcept {
  return data_.data();
}

typename String::iterator String::end() {
  return data_.mutableData() + data_.size();
}

typename String::const_iterator String::end() const noexcept {
  return data_.data() + data_.size();
}

typename String::const_iterator String::cend() const noexcept {
  return data_.data() + data_.size();
}

typename String::reverse_iterator String::rbegin() {
  return reverse_iterator(end());
}

typename String::const_reverse_iterator String::rbegin() const noexcept {
  return const_reverse_iterator(end());
}

typename String::const_reverse_iterator String::crbegin() const noexcept {
  return const_reverse_iterator(end());
}

typename String::reverse_iterator String::rend() {
  return reverse_iterator(begin());
}

typename String::const_reverse_iterator String::rend() const noexcept {
  return const_reverse_iterator(begin());
}

typename String::const_reverse_iterator String::crend() const noexcept {
  return const_reverse_iterator(begin());
}

}  // namespace runtime
}  // namespace matxscript
