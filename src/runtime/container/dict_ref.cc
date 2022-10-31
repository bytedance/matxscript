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
#include <matxscript/runtime/container/dict_ref.h>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/dict_private.h>
#include <matxscript/runtime/container/ft_dict.h>
#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/str_escape.h>

namespace matxscript {
namespace runtime {

template <>
inline RTValue IteratorSetRuntimeValue(Dict::item_iterator result) {
  return Tuple::dynamic(result->first, result->second);
}

class DictKeyIteratorNode : public IteratorNode {
 public:
  explicit DictKeyIteratorNode(Dict container,
                               Dict::container_type::iterator iter,
                               Dict::container_type::iterator iter_end)
      : container_(std::move(container)), first_(iter), last_(iter_end) {
  }
  ~DictKeyIteratorNode() = default;

  bool HasNext() const override {
    return first_ != last_;
  }
  RTValue Next() override {
    return (first_++)->first;
  }
  RTValue Next(bool* has_next) override {
    auto ret = (first_++)->first;
    *has_next = (first_ != last_);
    return ret;
  }
  RTView NextView(bool* has_next, RTValue* holder_or_null) override {
    RTView ret = (first_++)->first;
    *has_next = (first_ != last_);
    return ret;
  }

  int64_t Distance() const override {
    return std::distance(first_, last_);
  }

  uint64_t HashCode() const override {
    return reinterpret_cast<uint64_t>(container_.get());
  }

 public:
  Dict container_;
  Dict::container_type::iterator first_;
  Dict::container_type::iterator last_;
  friend class IteratorNodeTrait;
};

class DictValueIteratorNode : public IteratorNode {
 public:
  explicit DictValueIteratorNode(Dict container,
                                 Dict::container_type::iterator iter,
                                 Dict::container_type::iterator iter_end)
      : container_(std::move(container)), first_(iter), last_(iter_end) {
  }
  ~DictValueIteratorNode() = default;

  bool HasNext() const override {
    return first_ != last_;
  }
  RTValue Next() override {
    return (first_++)->second;
  }
  RTValue Next(bool* has_next) override {
    auto ret = (first_++)->second;
    *has_next = (first_ != last_);
    return ret;
  }
  RTView NextView(bool* has_next, RTValue* holder_or_null) override {
    RTView ret = (first_++)->second;
    *has_next = (first_ != last_);
    return ret;
  }
  int64_t Distance() const override {
    return std::distance(first_, last_);
  }

  uint64_t HashCode() const override {
    return reinterpret_cast<uint64_t>(container_.get());
  }

 public:
  Dict container_;
  Dict::container_type::iterator first_;
  Dict::container_type::iterator last_;
  friend class IteratorNodeTrait;
};

class DictItemIteratorNode : public IteratorNode {
 public:
  explicit DictItemIteratorNode(Dict container,
                                Dict::container_type::iterator iter,
                                Dict::container_type::iterator iter_end)
      : container_(std::move(container)), first_(iter), last_(iter_end) {
  }
  ~DictItemIteratorNode() = default;

  bool HasNext() const override {
    return first_ != last_;
  }
  RTValue Next() override {
    auto it = first_++;
    return Tuple::dynamic(it->first, it->second);
  }
  RTValue Next(bool* has_next) override {
    auto it = first_++;
    *has_next = (first_ != last_);
    return Tuple::dynamic(it->first, it->second);
  }
  RTView NextView(bool* has_next, RTValue* holder_or_null) override {
    auto it = first_++;
    *has_next = (first_ != last_);
    *holder_or_null = Tuple::dynamic(it->first, it->second);
    return *holder_or_null;
  }
  int64_t Distance() const override {
    return std::distance(first_, last_);
  }

  uint64_t HashCode() const override {
    return reinterpret_cast<uint64_t>(container_.get());
  }

 public:
  Dict container_;
  Dict::container_type::iterator first_;
  Dict::container_type::iterator last_;
  friend class IteratorNodeTrait;
};

Iterator Dict::item_iter() const {
  MX_CHECK_DPTR(Dict);
  auto data =
      make_object<DictItemIteratorNode>(*this, d->data_container.begin(), d->data_container.end());
  return Iterator(std::move(data));
}

Iterator Dict::key_iter() const {
  MX_CHECK_DPTR(Dict);
  auto data =
      make_object<DictKeyIteratorNode>(*this, d->data_container.begin(), d->data_container.end());
  return Iterator(std::move(data));
}

Iterator Dict::value_iter() const {
  MX_CHECK_DPTR(Dict);
  auto data =
      make_object<DictValueIteratorNode>(*this, d->data_container.begin(), d->data_container.end());
  return Iterator(std::move(data));
}

/******************************************************************************
 * Dict container
 *****************************************************************************/

void Dict::Init(const FuncGetNextItemRandom& func, size_t len) {
  auto node = make_object<DictNode>();
  node->reserve(len);
  for (size_t i = 0; i < len; ++i) {
    const auto& value = func();
    node->emplace(value.first, value.second);
  }
  data_ = std::move(node);
}

void Dict::Init(const FuncGetNextItemForward& func, bool has_next) {
  auto node = make_object<DictNode>();
  node->reserve(4);
  while (has_next) {
    const auto& value = func(has_next);
    node->emplace(value.first, value.second);
  }
  data_ = std::move(node);
}

Dict::Dict() {
  auto n = make_object<DictNode>();
  data_ = std::move(n);
}

Dict::Dict(Dict&& other) noexcept : ObjectRef() {  // NOLINT(*)
  data_ = std::move(other.data_);
}

Dict::Dict(const Dict& other) noexcept : ObjectRef() {  // NOLINT(*)
  data_ = other.data_;
}

Dict::Dict(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
}

Dict::Dict(std::initializer_list<value_type> init) {  // NOLINT(*)
  data_ = make_object<DictNode>(init.begin(), init.end());
}

Dict::Dict(const std::vector<value_type>& init) {  // NOLINT(*)
  data_ = make_object<DictNode>(init.begin(), init.end());
}

bool Dict::operator==(const Dict& other) const {
  auto* lhs_node = static_cast<DictNode*>(data_.get());
  auto* rhs_node = static_cast<DictNode*>(other.data_.get());
  if (lhs_node == rhs_node) {
    return true;
  }
  if (lhs_node->data_container.size() != rhs_node->data_container.size()) {
    return false;
  }
  if (lhs_node->data_container.size() == 0) {
    return true;
  }
  for (const auto& kv : lhs_node->data_container) {
    auto it = rhs_node->data_container.find(kv.first);
    if (it == rhs_node->data_container.end()) {
      return false;
    }
    if (!Any::Equal(kv.second, it->second)) {
      return false;
    }
  }
  return true;
}

bool Dict::operator!=(const Dict& other) const {
  return !operator==(other);
}

Dict::mapped_type& Dict::get_item(const Any& key) const {
  MX_CHECK_DPTR(Dict);
  auto iter = d->data_container.find(key);
  MXCHECK(iter != d->data_container.end()) << "Dict[" << key << "] not found";
  return iter->second;
}

Dict::mapped_type& Dict::get_item(const string_view& key) const {
  MX_CHECK_DPTR(Dict);
  auto iter = d->data_container.find(key);
  MXCHECK(iter != d->data_container.end()) << "Dict[" << key << "] not found";
  return iter->second;
}

Dict::mapped_type& Dict::get_item(const unicode_view& key) const {
  MX_CHECK_DPTR(Dict);
  auto iter = d->data_container.find(key);
  MXCHECK(iter != d->data_container.end()) << "Dict[" << key << "] not found";
  return iter->second;
}

Dict::mapped_type& Dict::get_item(int64_t key) const {
  MX_CHECK_DPTR(Dict);
  auto iter = d->data_container.find(key);
  MXCHECK(iter != d->data_container.end()) << "Dict[" << key << "] not found";
  return iter->second;
}

Dict::mapped_type const& Dict::get_default(const Any& key, mapped_type const& default_val) const {
  auto node = GetDictNode();
  if (node == nullptr) {
    return default_val;
  }
  auto it = node->data_container.find(key);
  return it == node->data_container.end() ? default_val : it->second;
}

Dict::mapped_type const& Dict::get_default(const string_view& key,
                                           Dict::mapped_type const& default_val) const {
  auto node = GetDictNode();
  if (node == nullptr) {
    return default_val;
  }
  auto it = node->data_container.find(key);
  return it == node->data_container.end() ? default_val : it->second;
}

Dict::mapped_type const& Dict::get_default(const unicode_view& key,
                                           Dict::mapped_type const& default_val) const {
  auto node = GetDictNode();
  if (node == nullptr) {
    return default_val;
  }
  auto it = node->data_container.find(key);
  return it == node->data_container.end() ? default_val : it->second;
}

Dict::mapped_type Dict::pop(PyArgs args) const {
  MXCHECK(args.size() == 1 || args.size() == 2)
      << "dict.pop expect 1 or 2 arguments, but get " << args.size();
  MX_DPTR(Dict);
  if (d == nullptr) {
    if (args.size() == 2) {
      return args[1].As<mapped_type>();
    }
    MXTHROW << "dict.pop KeyError";
  }
  auto it = d->data_container.find(args[0]);
  if (it == d->data_container.end()) {
    if (args.size() == 2) {
      return args[1].As<mapped_type>();
    }
    MXTHROW << "dict.pop KeyError";
  }
  auto ret = std::move(it->second);
  d->data_container.erase(it);
  return ret;
}

void Dict::set_item(key_type&& key, mapped_type&& value) const {
  MX_CHECK_DPTR(Dict);
  d->data_container[std::move(key)] = std::move(value);
}

Dict::mapped_type& Dict::operator[](Dict::key_type key) const {
  MX_CHECK_DPTR(Dict);
  return d->data_container[std::move(key)];
}

Dict::mapped_type& Dict::operator[](const char* key) const {
  MX_CHECK_DPTR(Dict);
  return d->data_container[String(key)];
}

Dict::mapped_type& Dict::operator[](const char32_t* key) const {
  MX_CHECK_DPTR(Dict);
  return d->data_container[Unicode(key)];
}

void Dict::emplace(Dict::key_type&& key, Dict::mapped_type&& value) const {
  MX_CHECK_DPTR(Dict);
  d->data_container.emplace(std::move(key), std::move(value));
}

void Dict::emplace(Dict::value_type&& value) const {
  MX_CHECK_DPTR(Dict);
  d->data_container.emplace(std::move(value));
}

void Dict::clear() const {
  MX_CHECK_DPTR(Dict);
  d->data_container.clear();
}

void Dict::reserve(int64_t new_size) const {
  if (new_size > 0) {
    MX_CHECK_DPTR(Dict);
    d->data_container.reserve(static_cast<size_t>(new_size));
  }
}

int64_t Dict::size() const {
  auto n = GetDictNode();
  return n == nullptr ? 0 : n->data_container.size();
}

int64_t Dict::bucket_count() const {
  auto n = GetDictNode();
  return n == nullptr ? 0 : n->data_container.bucket_count();
}

bool Dict::empty() const {
  return size() == 0;
}

bool Dict::contains(const Any& key) const {
  auto n = GetDictNode();
  return n == nullptr ? false : n->data_container.find(key) != n->data_container.end();
}

bool Dict::contains(const string_view& key) const {
  auto n = GetDictNode();
  return n == nullptr ? false : n->data_container.find(key) != n->data_container.end();
}

bool Dict::contains(const unicode_view& key) const {
  auto n = GetDictNode();
  return n == nullptr ? false : n->data_container.find(key) != n->data_container.end();
}

bool Dict::contains(int64_t key) const {
  auto n = GetDictNode();
  return n == nullptr ? false : n->data_container.find(key) != n->data_container.end();
}

DictItems<Dict> Dict::items() const {
  return DictItems<Dict>(*this);
}

DictKeys<Dict> Dict::keys() const {
  return DictKeys<Dict>(*this);
}

DictValues<Dict> Dict::values() const {
  return DictValues<Dict>(*this);
}

DictNode* Dict::GetDictNode() const {
  return static_cast<DictNode*>(data_.get());
}

typename Dict::iterator Dict::begin() const {
  auto n = GetDictNode();
  return typename Dict::iterator(n->data_container.begin());
}

typename Dict::iterator Dict::end() const {
  auto n = GetDictNode();
  return typename Dict::iterator(n->data_container.end());
}

typename Dict::item_iterator Dict::item_begin() const {
  auto n = GetDictNode();
  MXCHECK(n != nullptr) << "Dict container is null";
  return item_iterator_adaptator<typename Dict::container_type::iterator>(
      n->data_container.begin());
}

typename Dict::item_iterator Dict::item_end() const {
  auto n = GetDictNode();
  MXCHECK(n != nullptr) << "Dict.item_end container is null";
  return item_iterator_adaptator<typename Dict::container_type::iterator>(n->data_container.end());
}

typename Dict::key_const_iterator Dict::key_begin() const {
  auto n = GetDictNode();
  MXCHECK(n != nullptr) << "Dict.key_begin container is null";
  return key_iterator_adaptator<typename Dict::container_type::iterator>(n->data_container.begin());
}

typename Dict::key_const_iterator Dict::key_end() const {
  auto n = GetDictNode();
  MXCHECK(n != nullptr) << "Dict.key_end container is null";
  return key_iterator_adaptator<typename Dict::container_type::iterator>(n->data_container.end());
}

typename Dict::value_iterator Dict::value_begin() const {
  auto n = GetDictNode();
  MXCHECK(n != nullptr) << "Dict.value_begin container is null";
  return value_iterator_adaptator<typename Dict::container_type::iterator>(
      n->data_container.begin());
}

typename Dict::value_iterator Dict::value_end() const {
  auto n = GetDictNode();
  MXCHECK(n != nullptr) << "Dict.value_end container is null";
  return value_iterator_adaptator<typename Dict::container_type::iterator>(n->data_container.end());
}

std::ostream& operator<<(std::ostream& os, Dict const& n) {
  os << '{';
  for (auto it = n.begin(); it != n.end(); ++it) {
    if (it != n.begin()) {
      os << ", ";
    }
    if (it->first.IsString()) {
      auto view = it->first.AsNoCheck<string_view>();
      os << "b'" << BytesEscape(view.data(), view.size()) << "': ";
    } else if (it->first.IsUnicode()) {
      os << "\'" << it->first.As<unicode_view>() << "\': ";
    } else {
      os << it->first;
      os << ": ";
    }

    if (it->second.IsString()) {
      auto view = it->second.AsNoCheck<string_view>();
      os << "b'" << BytesEscape(view.data(), view.size()) << "'";
    } else if (it->second.IsUnicode()) {
      os << "\'" << it->second.As<unicode_view>() << "\'";
    } else {
      os << it->second;
    }
  }
  os << '}';
  return os;
}

template <>
bool IsConvertible<Dict>(const Object* node) {
  return node ? node->IsInstance<Dict::ContainerType>() : Dict::_type_is_nullable;
}

}  // namespace runtime
}  // namespace matxscript
