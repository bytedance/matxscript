// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 * https://github.com/apache/tvm/blob/v0.7/include/tvm/runtime/container.h
 * with changes applied:
 * - rename namespace
 * - implement some tuple methods
 *
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
#include <matxscript/runtime/container/tuple_ref.h>

#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/container/tuple_private.h>
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Generic Tuple Iterator
 *****************************************************************************/

class TupleIteratorNode : public IteratorNode {
 public:
  explicit TupleIteratorNode(Tuple container, Tuple::iterator iter, Tuple::iterator iter_end)
      : container_(std::move(container)), first_(iter), last_(iter_end) {
  }
  ~TupleIteratorNode() = default;

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
    return last_ - first_;
  }

  uint64_t HashCode() const override {
    return reinterpret_cast<uint64_t>(container_.get());
  }

 public:
  Tuple container_;
  Tuple::iterator first_;
  Tuple::iterator last_;
  friend class IteratorNodeTrait;
};

// iterators
Iterator Tuple::iter() const {
  auto data = make_object<TupleIteratorNode>(*this, begin(), end());
  return Iterator(std::move(data));
}

Tuple::iterator Tuple::begin() const {
  MX_CHECK_DPTR(Tuple);
  return d->begin();
}

Tuple::iterator Tuple::end() const {
  MX_CHECK_DPTR(Tuple);
  return d->end();
}

Tuple::reverse_iterator Tuple::rbegin() const {
  MX_CHECK_DPTR(Tuple);
  return d->rbegin();
}

Tuple::reverse_iterator Tuple::rend() const {
  MX_CHECK_DPTR(Tuple);
  return d->rend();
}

/******************************************************************************
 * Tuple container
 *****************************************************************************/

template <>
bool IsConvertible<Tuple>(const Object* node) {
  return node ? node->IsInstance<Tuple::ContainerType>() : Tuple::_type_is_nullable;
}

Tuple Tuple::Empty(size_t capacity) {
  auto ptr = make_inplace_array_object<TupleNode, value_type>(capacity);
  ptr->size = 0;
  return Tuple(std::move(ptr));
}

void Tuple::AllocN(size_t len) {
  auto ptr = make_inplace_array_object<TupleNode, value_type>(len);
  ptr->size = 0;
  data_ = std::move(ptr);
}

Tuple& Tuple::EmplaceUnsafe(value_type&& ele) {
  MX_DPTR(Tuple);
  d->EmplaceInit(d->size, std::move(ele));
  // Only increment size after the initialization succeeds
  d->size++;
  return *this;
}

void Tuple::Init(const FuncGetNextItem& func, size_t len) {
  auto ptr = make_inplace_array_object<TupleNode, value_type>(len);
  ptr->size = 0;
  for (size_t i = 0; i < len; ++i) {
    ptr->EmplaceInit(i, func());
    // Only increment size after the initialization succeeds
    ptr->size++;
  }
  data_ = std::move(ptr);
}

Tuple::Tuple(const Any* begin, const Any* end) {
  auto num = std::distance(begin, end);
  auto ptr = make_inplace_array_object<TupleNode, value_type>(num);
  ptr->size = 0;
  for (size_t i = 0; i < num; ++i) {
    ptr->EmplaceInit(i, begin->As<RTValue>());
    // Only increment size after the initialization succeeds
    ptr->size++;
  }
  data_ = std::move(ptr);
}

Tuple::Tuple(std::initializer_list<value_type> init) {
  auto ptr = make_inplace_array_object<TupleNode, value_type>(init.size());
  ptr->Init(init.begin(), init.end());
  data_ = std::move(ptr);
}

bool Tuple::operator==(const Tuple& other) const {
  auto* lhs_node = static_cast<TupleNode*>(data_.get());
  auto* rhs_node = static_cast<TupleNode*>(other.data_.get());
  if (lhs_node == rhs_node) {
    return true;
  }
  if (lhs_node->size != rhs_node->size) {
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

bool Tuple::operator>(const Tuple& other) const {
  auto* lhs_node = static_cast<TupleNode*>(data_.get());
  auto* rhs_node = static_cast<TupleNode*>(other.data_.get());

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

bool Tuple::operator>=(const Tuple& other) const {
  auto* lhs_node = static_cast<TupleNode*>(data_.get());
  auto* rhs_node = static_cast<TupleNode*>(other.data_.get());

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

Tuple::value_type& Tuple::operator[](size_t idx) const {
  MX_CHECK_DPTR(Tuple);
  return d->operator[](idx);
}

Tuple::value_type& Tuple::get_item(int64_t idx) const {
  MX_CHECK_DPTR(Tuple);
  return d->operator[](idx);
}

int64_t Tuple::size() const {
  MX_DPTR(Tuple);
  return d->size;
}

Tuple Tuple::repeat(int64_t times) const {
  MX_DPTR(Tuple);
  auto new_size = d->size * times;
  auto new_node = make_inplace_array_object<TupleNode, value_type>(new_size);
  new_node->size = 0;
  int k = 0;
  for (auto t = 0; t < times; ++t) {
    for (size_t i = 0; i < d->size; ++i) {
      new_node->EmplaceInit(k++, *reinterpret_cast<value_type*>(d->AddressOf(i)));
      // Only increment size after the initialization succeeds
      new_node->size++;
    }
  }
  return Tuple(std::move(new_node));
}

Tuple Tuple::get_slice(int64_t b, int64_t e, int64_t step) const {
  MXCHECK_GT(step, 0) << "Tuple.slice_load step must be gt 0";
  MX_DPTR(Tuple);
  int64_t len = size();
  b = slice_index_correction(b, len);
  e = slice_index_correction(e, len);
  if (e <= b) {
    return Tuple();
  } else {
    if (step == 1) {
      auto new_size = e - b;
      auto new_node = make_inplace_array_object<TupleNode, value_type>(new_size);
      new_node->size = 0;
      int64_t k = 0;
      for (int64_t i = b; i < e; ++i) {
        new_node->EmplaceInit(k++, *reinterpret_cast<value_type*>(d->AddressOf(i)));
        // Only increment size after the initialization succeeds
        new_node->size++;
      }
      return Tuple(std::move(new_node));
    } else {
      auto new_size = (e - b + step - 1) / step;
      auto new_node = make_inplace_array_object<TupleNode, value_type>(new_size);
      new_node->size = 0;
      int64_t k = 0;
      for (int64_t i = b; i < e; i += step) {
        new_node->EmplaceInit(k++, *reinterpret_cast<value_type*>(d->AddressOf(i)));
        // Only increment size after the initialization succeeds
        new_node->size++;
      }
      return Tuple(std::move(new_node));
    }
  }
}

bool Tuple::find_match_fn(const FuncEqualToValue& fn) const {
  MX_DPTR(Tuple);
  if (!d) {
    return false;
  }
  return std::any_of(d->begin(), d->end(), fn);
}

int64_t Tuple::count_match_fn(const FuncEqualToValue& fn) const {
  MX_DPTR(Tuple);
  if (!d) {
    return false;
  }
  int64_t cou = 0;
  for (auto& item : *d) {
    if (fn(item)) {
      ++cou;
    }
  }
  return cou;
}

Tuple Tuple::Concat(Tuple lhs, Tuple rhs) {
  auto* lhs_node = static_cast<const TupleNode*>(lhs.get());
  auto* rhs_node = static_cast<const TupleNode*>(rhs.get());
  auto new_size = lhs_node->size + rhs_node->size;
  auto new_node = make_inplace_array_object<TupleNode, Tuple::value_type>(new_size);
  new_node->size = 0;
  int64_t k = 0;
  if (lhs.use_count() == 1) {
    for (size_t i = 0; i < lhs_node->size; ++i) {
      new_node->EmplaceInit(k++, std::move(*reinterpret_cast<value_type*>(lhs_node->AddressOf(i))));
      // Only increment size after the initialization succeeds
      new_node->size++;
    }
  } else {
    for (size_t i = 0; i < lhs_node->size; ++i) {
      new_node->EmplaceInit(k++, *reinterpret_cast<value_type*>(lhs_node->AddressOf(i)));
      // Only increment size after the initialization succeeds
      new_node->size++;
    }
  }

  if (rhs.use_count() == 1) {
    for (size_t i = 0; i < rhs_node->size; ++i) {
      new_node->EmplaceInit(k++, std::move(*reinterpret_cast<value_type*>(rhs_node->AddressOf(i))));
      // Only increment size after the initialization succeeds
      new_node->size++;
    }
  } else {
    for (size_t i = 0; i < rhs_node->size; ++i) {
      new_node->EmplaceInit(k++, *reinterpret_cast<value_type*>(rhs_node->AddressOf(i)));
      // Only increment size after the initialization succeeds
      new_node->size++;
    }
  }
  return Tuple(std::move(new_node));
}

std::ostream& operator<<(std::ostream& os, Tuple const& n) {
  auto* op = static_cast<const TupleNode*>(n.get());
  os << "(";
  for (size_t i = 0; i < op->size; ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << n[i];
  }
  os << ')';
  return os;
}

}  // namespace runtime
}  // namespace matxscript
