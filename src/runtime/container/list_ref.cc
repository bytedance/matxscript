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
#include <matxscript/runtime/container/list_ref.h>

#include <algorithm>

#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/ft_list.h>
#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/container/list_helper.h>
#include <matxscript/runtime/container/list_private.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/str_escape.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Generic List Iterator
 *****************************************************************************/

template <typename ListBiDirectionalIterator>
class ListIteratorNode : public IteratorNode {
 public:
  explicit ListIteratorNode(List container,
                            ListBiDirectionalIterator iter,
                            ListBiDirectionalIterator iter_end)
      : container_(std::move(container)), first_(iter), last_(iter_end) {
  }
  ~ListIteratorNode() = default;

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
  List container_;
  ListBiDirectionalIterator first_;
  ListBiDirectionalIterator last_;
  friend class IteratorNodeTrait;
};

/******************************************************************************
 * List container
 *****************************************************************************/

void List::Init(const FuncGetNextItemRandom& func, size_t len) {
  auto node = make_object<ListNode>();
  node->reserve(len);
  for (size_t i = 0; i < len; ++i) {
    node->emplace_back(func());
  }
  data_ = std::move(node);
}

void List::Init(const FuncGetNextItemForward& func, bool has_next) {
  auto node = make_object<ListNode>();
  node->reserve(4);
  while (has_next) {
    node->emplace_back(func(has_next));
  }
  data_ = std::move(node);
}

List::List() {
  data_ = make_object<ListNode>();
}

List::List(const Any* begin, const Any* end) {
  auto node = make_object<ListNode>();
  node->reserve(std::distance(begin, end));
  for (; begin != end; ++begin) {
    node->emplace_back(begin->As<RTValue>());
  }
  data_ = std::move(node);
}

List::List(std::initializer_list<value_type> init) {
  data_ = make_object<ListNode>(init);
}

List::List(const std::vector<value_type>& init) {
  data_ = make_object<ListNode>(init.begin(), init.end());
}

List::List(size_t n, const value_type& val) {
  data_ = make_object<ListNode>(n, val);
}

ListNode* List::CreateOrGetListNode() {
  if (!data_.get()) {
    data_ = make_object<ListNode>();
  }
  return static_cast<ListNode*>(data_.get());
}

ListNode* List::GetListNode() const {
  return static_cast<ListNode*>(data_.get());
}

bool List::operator==(const List& other) const {
  auto* lhs_node = static_cast<ListNode*>(data_.get());
  auto* rhs_node = static_cast<ListNode*>(other.data_.get());
  if (lhs_node == rhs_node) {
    return true;
  }
  if (lhs_node->size() != rhs_node->size()) {
    return false;
  }
  auto rhs_itr = rhs_node->begin();
  for (auto lhs_itr = lhs_node->begin(); lhs_itr != lhs_node->end(); ++lhs_itr, ++rhs_itr) {
    if (!Any::Equal(*lhs_itr, *rhs_itr)) {
      return false;
    }
  }
  return true;
}

bool List::operator>(const List& other) const {
  auto* lhs_node = static_cast<ListNode*>(data_.get());
  auto* rhs_node = static_cast<ListNode*>(other.data_.get());

  auto l_it = lhs_node->begin();
  auto r_it = rhs_node->begin();
  for (; l_it != lhs_node->end() && r_it != rhs_node->end(); ++l_it, ++r_it) {
    if (ArithOps::gt(*l_it, *r_it)) {
      return true;
    } else if (ArithOps::gt(*r_it, *l_it)) {
      return false;
    }
  }
  if (l_it != lhs_node->end()) {
    return true;
  } else {
    return false;
  }
}

bool List::operator>=(const List& other) const {
  auto* lhs_node = static_cast<ListNode*>(data_.get());
  auto* rhs_node = static_cast<ListNode*>(other.data_.get());

  auto l_it = lhs_node->begin();
  auto r_it = rhs_node->begin();
  for (; l_it != lhs_node->end() && r_it != rhs_node->end(); ++l_it, ++r_it) {
    if (ArithOps::gt(*l_it, *r_it)) {
      return true;
    } else if (ArithOps::gt(*r_it, *l_it)) {
      return false;
    }
  }
  if (r_it == rhs_node->end()) {
    return true;
  } else {
    return false;
  }
}

List::value_type& List::operator[](int64_t i) const {
  MX_CHECK_DPTR(List);
  return d->operator[](i);
}

int64_t List::size() const {
  ListNode* p = GetListNode();
  return p == nullptr ? 0 : p->size();
}

bool List::find_match_fn(const FuncEqualToValue& fn) const {
  ListNode* node = GetListNode();
  if (!node) {
    return false;
  }
  return std::any_of(node->begin(), node->end(), fn);
}

int64_t List::find_match_idx_fn(const FuncEqualToValue& fn, int64_t start, int64_t end) const {
  ListNode* node = GetListNode();
  if (!node) {
    return false;
  }
  return std::find_if(node->begin() + start, node->begin() + end, fn) - node->begin();
}

int64_t List::count_match_fn(const FuncEqualToValue& fn) const {
  ListNode* node = GetListNode();
  if (!node) {
    return false;
  }
  int64_t cou = 0;
  for (auto& item : *node) {
    if (fn(item)) {
      ++cou;
    }
  }
  return cou;
}

int64_t List::capacity() const {
  ListNode* p = GetListNode();
  return p == nullptr ? 0 : p->capacity();
}

void List::push_back(List::value_type&& item) const {
  MX_CHECK_DPTR(List);
  d->emplace_back(std::move(item));
}

void List::push_back(const List::value_type& item) const {
  MX_CHECK_DPTR(List);
  d->emplace_back(item);
}

void List::pop_back() const {
  MX_CHECK_DPTR(List);
  d->pop_back();
}

// python method
List::value_type& List::get_item(int64_t i) const {
  MX_CHECK_DPTR(List);
  int64_t len = size();
  MXCHECK((i >= 0 && i < len) || (i < 0 && i >= -len)) << "ValueError: index overflow";
  i = slice_index_correction(i, len);
  return d->data_container[i];
}

void List::set_item(int64_t i, value_type&& item) const {
  ListNode* p = GetListNode();
  int64_t len = (p == nullptr ? 0 : p->size());
  if (i < 0) {
    i += len;
  }
  MXCHECK(i >= 0 && i < len) << "ValueError: index overflow";
  p->data_container[i] = std::move(item);
}

void List::set_item(int64_t i, const value_type& item) const {
  return set_item(i, value_type(item));
}

List List::get_slice(int64_t b, int64_t e, int64_t step) const {
  MXCHECK_GT(step, 0) << "List.slice_load step must be gt 0";
  int64_t len = size();
  b = slice_index_correction(b, len);
  e = slice_index_correction(e, len);
  if (e <= b) {
    return List();
  } else {
    if (step == 1) {
      return List(this->begin() + b, this->begin() + e);
    } else {
      List new_list;
      new_list.reserve(e - b);
      auto itr_end = begin() + e;
      for (auto itr = begin() + b; itr < itr_end; itr += step) {
        new_list.push_back(*itr);
      }
      return new_list;
    }
  }
}

void List::set_slice(int64_t start, int64_t end, List&& rhs) const {
  MXCHECK(start >= 0 && end >= 0 && start <= end);
  int64_t len = size();
  start = slice_index_correction(start, len);
  end = slice_index_correction(end, len);
  ListNode* p = GetListNode();
  p->data_container.erase(p->data_container.begin() + start, p->data_container.begin() + end);
  if (rhs.use_count() == 1) {
    p->data_container.insert(p->data_container.begin() + start,
                             std::make_move_iterator(rhs.begin()),
                             std::make_move_iterator(rhs.end()));
  } else {
    p->data_container.insert(p->data_container.begin() + start, rhs.begin(), rhs.end());
  }
}

void List::set_slice(int64_t start, int64_t end, const List& rhs) const {
  MXCHECK(start >= 0 && end >= 0 && start <= end);
  int64_t len = size();
  start = slice_index_correction(start, len);
  end = slice_index_correction(end, len);
  ListNode* p = GetListNode();
  p->data_container.erase(p->data_container.begin() + start, p->data_container.begin() + end);
  p->data_container.insert(p->data_container.begin() + start, rhs.begin(), rhs.end());
}

void List::reserve(int64_t new_size) const {
  if (new_size > 0) {
    MX_CHECK_DPTR(List);
    d->reserve(static_cast<int>(new_size));
  }
}

void List::resize(int64_t new_size) const {
  if (new_size >= 0) {
    MX_CHECK_DPTR(List);
    d->resize(static_cast<int>(new_size));
  }
}

void List::extend(List&& items) const {
  MX_CHECK_DPTR(List);
  d->reserve(size() + items.size());
  if (items.use_count() == 1) {
    auto b = std::make_move_iterator(items.begin());
    auto e = std::make_move_iterator(items.end());
    for (; b != e; ++b) {
      d->emplace_back(std::move(*b));
    }
  } else {
    for (const auto& item : items) {
      d->emplace_back(item);
    }
  }
}

void List::extend(const List& items) const {
  MX_CHECK_DPTR(List);
  d->reserve(size() + items.size());
  for (const auto& item : items) {
    d->emplace_back(item);
  }
}

void List::extend(const Iterator& items) const {
  MX_CHECK_DPTR(List);
  bool has_next = items.HasNext();
  while (has_next) {
    d->emplace_back(items.Next(&has_next));
  }
}

void List::extend(const Any& items) const {
  switch (items.type_code()) {
    case TypeIndex::kRuntimeList: {
      this->extend(items.AsNoCheck<List>());
    } break;
    case TypeIndex::kRuntimeIterator: {
      this->extend(items.AsObjectViewNoCheck<Iterator>().data());
    } break;
    default: {
      this->extend(Kernel_Iterable::make(items));
    } break;
  }
}

namespace {
static inline constexpr bool CanUseFastCopy(const Any& v) {
  return v.type_code() == TypeIndex::kRuntimeInteger || v.type_code() == TypeIndex::kRuntimeFloat ||
         v.type_code() == TypeIndex::kRuntimeNullptr;
}
}  // namespace

List List::repeat(int64_t times) const {
  MX_CHECK_DPTR(List);

  List new_list{};
  if (MATXSCRIPT_UNLIKELY(times <= 0)) {
    return new_list;
  }

  auto new_node = new_list.GetListNode();
  new_node->reserve(times * d->size());
  auto this_b = d->begin();
  auto this_e = d->end();
  auto num_ele = this_e - this_b;

  // eval copy function and do copy
  if (MATXSCRIPT_UNLIKELY(num_ele == 0)) {
  } else if (MATXSCRIPT_LIKELY(num_ele == 1)) {
    const value_type& ele = *this_b;
    if (CanUseFastCopy(ele)) {
      for (int64_t i = 0; i < times; i++) {
        new_node->emplace_back(ele.value(), RTValue::ScalarValueFlag{});
      }
    } else {
      for (int64_t i = 0; i < times; i++) {
        new_node->emplace_back(ele);
      }
    }
  } else {
    for (int64_t i = 0; i < times; i++) {
      for (auto iter = this_b; iter != this_e; ++iter) {
        new_node->emplace_back(*iter);
      }
    }
  }

  return new_list;
}

List List::repeat_one(const Any& value, int64_t times) {
  List new_list{};
  if (MATXSCRIPT_UNLIKELY(times <= 0)) {
    return new_list;
  }
  auto new_node = new_list.GetListNode();
  new_node->reserve(times);
  if (CanUseFastCopy(value)) {
    for (int64_t i = 0; i < times; i++) {
      new_node->emplace_back(value.value(), RTValue::ScalarValueFlag{});
    }
  } else {
    for (int64_t i = 0; i < times; i++) {
      new_node->emplace_back(value);
    }
  }
  return new_list;
}

List List::repeat_one(value_type&& value, int64_t times) {
  List new_list{};
  if (MATXSCRIPT_UNLIKELY(times <= 0)) {
    return new_list;
  }
  auto new_node = new_list.GetListNode();
  new_node->reserve(times);
  times = times - 1;
  if (CanUseFastCopy(value)) {
    for (int64_t i = 0; i < times; i++) {
      new_node->emplace_back(value.value(), RTValue::ScalarValueFlag{});
    }
  } else {
    for (int64_t i = 0; i < times; i++) {
      new_node->emplace_back(value);
    }
  }
  new_node->emplace_back(std::move(value));
  return new_list;
}

List List::repeat_many(const std::initializer_list<value_type>& values, int64_t times) {
  List new_list{};
  if (MATXSCRIPT_UNLIKELY(times <= 0)) {
    return new_list;
  }
  auto new_node = new_list.GetListNode();
  new_node->reserve(times * values.size());
  auto this_b = values.begin();
  auto this_e = values.end();
  auto num_ele = this_e - this_b;
  for (int64_t i = 0; i < times; i++) {
    for (auto iter = this_b; iter != this_e; ++iter) {
      new_node->emplace_back(*iter);
    }
  }
  return new_list;
}

void List::clear() const {
  MX_CHECK_DPTR(List);
  d->clear();
}

void List::remove(const Any& item) const {
  MX_CHECK_DPTR(List);
  d->remove(item);
}

List::value_type List::pop(int64_t index) const {
  MX_CHECK_DPTR(List);
  return d->pop(index);
}

void List::insert(int64_t index, const Any& item) const {
  MX_CHECK_DPTR(List);
  d->insert(index, item);
}

void List::insert(int64_t index, List::value_type&& item) const {
  MX_CHECK_DPTR(List);
  d->insert(index, std::move(item));
}

void List::reverse() const {
  MX_CHECK_DPTR(List);
  return d->reverse();
}

void List::sort(bool reverse) const {
  if (reverse) {
    auto reverse_func = [](const RTValue& lhs, const RTValue& rhs) {
      return ArithOps::ge(lhs, rhs);
    };
    sort::pdqsort(this->begin(), this->end(), reverse_func);
  } else {
    sort::pdqsort(this->begin(), this->end(), ArithOps::lt<const RTValue&, const RTValue&>);
  }
}

void List::sort(const Any& key, bool reverse) const {
  if (!key.IsObjectRef<UserDataRef>()) {
    THROW_PY_TypeError("'", key.type_name(), "' object is not callable");
  }
  auto key_func = key.AsObjectRefNoCheck<UserDataRef>();
  if (reverse) {
    auto reverse_func = [&key_func](const RTValue& lhs, const RTValue& rhs) {
      return ArithOps::ge(key_func.generic_call(PyArgs(&lhs, 1)),
                          key_func.generic_call(PyArgs(&rhs, 1)));
    };
    sort::pdqsort(this->begin(), this->end(), reverse_func);
  } else {
    auto func = [&key_func](const RTValue& lhs, const RTValue& rhs) {
      return ArithOps::lt<const RTValue&, const RTValue&>(key_func.generic_call(PyArgs(&lhs, 1)),
                                                          key_func.generic_call(PyArgs(&rhs, 1)));
    };
    sort::pdqsort(this->begin(), this->end(), func);
  }
}

// iterators
Iterator List::iter() const {
  auto data = make_object<ListIteratorNode<List::iterator>>(*this, begin(), end());
  return Iterator(std::move(data));
}

List::iterator List::begin() const {
  MX_CHECK_DPTR(List);
  return d->begin();
}

List::iterator List::nocheck_begin() const {
  MX_DPTR(List);
  return d->begin();
}

List::iterator List::end() const {
  MX_CHECK_DPTR(List);
  return d->end();
}

List::iterator List::nocheck_end() const {
  MX_DPTR(List);
  return d->end();
}

List::reverse_iterator List::rbegin() const {
  MX_CHECK_DPTR(List);
  return d->rbegin();
}

List::reverse_iterator List::nocheck_rbegin() const {
  MX_DPTR(List);
  return d->rbegin();
}

List::reverse_iterator List::rend() const {
  MX_CHECK_DPTR(List);
  return d->rend();
}

List::reverse_iterator List::nocheck_rend() const {
  MX_DPTR(List);
  return d->rend();
}

List::value_type* List::data() const {
  MX_DPTR(List);
  return d ? d->data_container.data() : nullptr;
}

// construct method
template <>
List List::Concat<true>(std::initializer_list<List> data) {
  size_t cap = 0;
  for (auto& con : data) {
    cap += con.size();
  }
  if (cap <= 0) {
    return List{};
  }
  auto itr = data.begin();
  auto itr_end = data.end();
  ObjectPtr<ListNode> result_node{nullptr};
  if (itr->use_count() == 1) {
    result_node = GetObjectPtr<ListNode>(static_cast<ListNode*>(itr->data_.get()));
    ++itr;
  } else {
    result_node = make_object<ListNode>();
  }
  result_node->reserve(cap);
  for (; itr != itr_end; ++itr) {
    if (itr->use_count() == 1) {
      auto mov_b = std::make_move_iterator(itr->begin());
      auto mov_e = std::make_move_iterator(itr->end());
      for (; mov_b != mov_e; ++mov_b) {
        result_node->emplace_back(*mov_b);
      }
    } else {
      for (auto& x : *itr) {
        result_node->emplace_back(x);
      }
    }
  }
  return List{std::move(result_node)};
}

template <>
List List::Concat<false>(std::initializer_list<List> data) {
  ObjectPtr<ListNode> result_node = make_object<ListNode>();
  size_t cap = 0;
  for (auto& con : data) {
    cap += con.size();
  }
  if (cap <= 0) {
    return List{std::move(result_node)};
  }
  result_node->reserve(cap);
  for (auto& cons : data) {
    for (auto& x : cons) {
      result_node->emplace_back(x);
    }
  }
  return List{result_node};
}

Iterator List::builtins_iter(const List& iterable) {
  auto data =
      make_object<ListIteratorNode<List::iterator>>(iterable, iterable.begin(), iterable.end());
  return Iterator(std::move(data));
}

Iterator List::builtins_reversed(const List& iterable) {
  auto data = make_object<ListIteratorNode<List::reverse_iterator>>(
      iterable, iterable.rbegin(), iterable.rend());
  return Iterator(std::move(data));
}

template <>
bool IsConvertible<List>(const Object* node) {
  return node ? node->IsInstance<List::ContainerType>() : List::_type_is_nullable;
}

std::ostream& operator<<(std::ostream& os, List const& n) {
  auto* op = static_cast<const ListNode*>(n.get());
  List obj = GetRef<List>(op);
  os << '[';
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    if (it != obj.begin()) {
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
  os << ']';
  return os;
}

}  // namespace runtime
}  // namespace matxscript
