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
#include <matxscript/runtime/container/set_ref.h>

#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/container/set_private.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/str_escape.h>

namespace matxscript {
namespace runtime {

class SetIteratorNode : public IteratorNode {
 public:
  explicit SetIteratorNode(Set container, Set::const_iterator iter, Set::const_iterator iter_end)
      : container_(std::move(container)), first_(iter), last_(iter_end) {
  }
  ~SetIteratorNode() = default;

  bool HasNext() const override {
    return first_ != last_;
  }
  RTValue Next() override {
    return *(first_++);
  }
  RTValue Next(bool* has_next) override {
    auto ret = *(first_++);
    *has_next = (first_ != last_);
    return ret;
  }
  RTView NextView(bool* has_next, RTValue* holder_or_null) override {
    RTView ret = *(first_++);
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
  Set container_;
  Set::const_iterator first_;
  Set::const_iterator last_;
};

Iterator Set::iter() const {
  auto data = make_object<SetIteratorNode>(*this, begin(), end());
  return Iterator(std::move(data));
}

/******************************************************************************
 * Set container
 *****************************************************************************/

void Set::Init(const FuncGetNextItemRandom& func, size_t len) {
  auto node = make_object<SetNode>();
  node->data_container.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    node->data_container.emplace(func());
  }
  data_ = std::move(node);
}

void Set::Init(const FuncGetNextItemForward& func, bool has_next) {
  auto node = make_object<SetNode>();
  node->reserve(4);
  while (has_next) {
    node->emplace(func(has_next));
  }
  data_ = std::move(node);
}

Set::Set() {
  data_ = make_object<SetNode>();
}

Set::Set(Set&& other) noexcept : ObjectRef() {  // NOLINT(*)
  data_ = std::move(other.data_);
}

Set::Set(const Set& other) noexcept : ObjectRef() {  // NOLINT(*)
  data_ = other.data_;
}

Set::Set(const Any* begin, const Any* end) {
  auto node = make_object<SetNode>();
  node->reserve(std::distance(begin, end));
  for (; begin != end; ++begin) {
    node->emplace(begin->As<RTValue>());
  }
  data_ = std::move(node);
}

Set::Set(std::initializer_list<value_type> init) {  // NOLINT(*)
  data_ = make_object<SetNode>(init.begin(), init.end());
}

Set::Set(const std::vector<value_type>& init) {  // NOLINT(*)
  data_ = make_object<SetNode>(init.begin(), init.end());
}

Set& Set::operator=(Set&& other) noexcept {
  data_ = std::move(other.data_);
  return *this;
}

Set& Set::operator=(const Set& other) noexcept {
  data_ = other.data_;
  return *this;
}

bool Set::operator==(const Set& other) const {
  auto* lhs_node = static_cast<SetNode*>(data_.get());
  auto* rhs_node = static_cast<SetNode*>(other.data_.get());
  if (lhs_node == rhs_node) {
    return true;
  }
  if (lhs_node->size() != rhs_node->size()) {
    return false;
  }
  if (lhs_node->empty()) {
    return true;
  }
  auto rhs_itr = rhs_node->begin();
  for (auto lhs_itr = lhs_node->begin(); lhs_itr != lhs_node->end(); ++lhs_itr, ++rhs_itr) {
    auto it = rhs_node->find(*lhs_itr);
    if (it == rhs_node->end()) {
      return false;
    }
  }
  return true;
}

bool Set::operator!=(const Set& other) const {
  return !operator==(other);
}

void Set::emplace(value_type&& item) const {
  MX_CHECK_DPTR(Set);
  d->emplace(std::move(item));
}

void Set::clear() const {
  MX_DPTR(Set);
  if (d) {
    d->clear();
  }
}

void Set::reserve(int64_t new_size) const {
  MX_DPTR(Set);
  d->reserve(new_size);
}

int64_t Set::size() const {
  auto n = GetSetNode();
  return n == nullptr ? 0 : n->size();
}

int64_t Set::bucket_count() const {
  auto n = GetSetNode();
  return n == nullptr ? 0 : n->bucket_count();
}

bool Set::empty() const {
  return size() == 0;
}

bool Set::contains(const Any& key) const {
  auto n = GetSetNode();
  return n == nullptr ? false : n->contains(key);
}

bool Set::contains(string_view key) const {
  auto n = GetSetNode();
  return n == nullptr ? false : n->contains(key);
}

bool Set::contains(unicode_view key) const {
  auto n = GetSetNode();
  return n == nullptr ? false : n->contains(key);
}

bool Set::contains(int64_t key) const {
  auto n = GetSetNode();
  return n == nullptr ? false : n->contains(key);
}

void Set::discard(const Any& rt_value) const {
  auto n = GetSetNode();
  if (n) {
    auto it = n->find(rt_value);
    if (it != n->end()) {
      (n->data_container).erase(it);
    }
  }
}

void Set::difference_update_iter(const Iterator& iter) const {
  if (iter.defined()) {
    auto iter_node = iter.GetMutableNode();
    while (iter_node->HasNext()) {
      this->discard(iter_node->Next());
    }
  }
}

void Set::update_iter(const Iterator& iter) const {
  if (iter.defined()) {
    auto iter_node = iter.GetMutableNode();
    while (iter_node->HasNext()) {
      this->add(iter_node->Next());
    }
  }
}

void Set::difference_update(PyArgs args) const {
  for (const auto* it = args.begin(); it != args.end(); ++it) {
    if (it->type_code() == TypeIndex::kRuntimeIterator) {
      this->difference_update_iter(it->AsObjectRefNoCheck<Iterator>());
    } else {
      this->difference_update_iter(Kernel_Iterable::make(*it));
    }
  }
}

Set Set::difference(PyArgs args) const {
  Set ret(make_object<SetNode>(*GetSetNode()));
  ret.difference_update(args);
  return ret;
}

void Set::update(PyArgs args) const {
  for (const auto* it = args.begin(); it != args.end(); ++it) {
    if (it->type_code() == TypeIndex::kRuntimeIterator) {
      this->update_iter(it->AsObjectRefNoCheck<Iterator>());
    } else {
      this->update_iter(Kernel_Iterable::make(*it));
    }
  }
}

Set Set::set_union(PyArgs args) const {
  Set ret(make_object<SetNode>(*GetSetNode()));
  ret.update(args);
  return ret;
}

SetNode* Set::CreateOrGetSetNode() {
  if (!data_.get()) {
    data_ = make_object<SetNode>();
  }
  return static_cast<SetNode*>(data_.get());
}

SetNode* Set::GetSetNode() const {
  return static_cast<SetNode*>(data_.get());
}

Set::const_iterator Set::begin() const {
  MX_CHECK_DPTR(Set);
  return d->begin();
}

Set::const_iterator Set::end() const {
  MX_CHECK_DPTR(Set);
  return d->end();
}

std::ostream& operator<<(std::ostream& os, Set const& n) {
  os << '{';
  for (auto it = n.begin(); it != n.end(); ++it) {
    if (it != n.begin()) {
      os << ", ";
    }
    if (it->IsString()) {
      auto view = it->AsNoCheck<string_view>();
      os << "b'" << BytesEscape(view.data(), view.size()) << "'";
    } else if (it->IsUnicode()) {
      os << "\'" << it->AsNoCheck<unicode_view>() << "\'";
    } else {
      os << *it;
    }
  }
  os << '}';
  return os;
}

template <>
bool IsConvertible<Set>(const Object* node) {
  return node ? node->IsInstance<Set::ContainerType>() : Set::_type_is_nullable;
}

}  // namespace runtime
}  // namespace matxscript
