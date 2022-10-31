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
#include <matxscript/runtime/container/itertor_ref.h>

#include <matxscript/runtime/container/_ft_object_base.h>
#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

Iterator Iterator::MakeGenericIterator(RTValue container,
                                       std::function<bool()> has_next,
                                       std::function<RTValue()> next,
                                       std::function<RTValue(bool*)> next_and_check) {
  auto data = make_object<GenericIteratorNode>(
      std::move(container), std::move(has_next), std::move(next), std::move(next_and_check));
  return Iterator(std::move(data));
}

Iterator Iterator::MakeGenericIterator(const Any& container) {
  return Kernel_Iterable::make(container);
}

Iterator Iterator::MakeItemsIterator(const Any& container) {
  if (container.type_code() == TypeIndex::kRuntimeDict) {
    return container.AsObjectViewNoCheck<Dict>().data().item_iter();
  } else if (container.type_code() == TypeIndex::kRuntimeFTDict) {
    return container.AsObjectViewNoCheck<FTObjectBase>()
        .data()
        .generic_call_attr("items", {})
        .As<Iterator>();
  } else {
    return Kernel_Iterable::make(container);
  }
}

bool Iterator::all_items_equal(const Iterator& lhs, const Iterator& rhs) {
  // maybe we need a deep copy
  auto lhs_node = lhs.GetMutableNode();
  auto rhs_node = rhs.GetMutableNode();
  if (lhs_node == rhs_node) {
    return true;
  }
  bool has_next_l = lhs_node->HasNext();
  bool has_next_r = rhs_node->HasNext();
  if (has_next_l != has_next_r) {
    return false;
  }
  while (has_next_l && has_next_r) {
    auto lhs_v = lhs_node->Next(&has_next_l);
    auto rhs_v = rhs_node->Next(&has_next_r);
    if (lhs_v != rhs_v) {
      return false;
    }
  }
  if (has_next_l != has_next_r) {
    return false;
  }
  return true;
}

template <>
bool IsConvertible<Iterator>(const Object* node) {
  return node ? node->IsInstance<Iterator::ContainerType>() : Iterator::_type_is_nullable;
}

IteratorNode* Iterator::GetMutableNode() const {
  MX_DPTR(Iterator);
  return d;
}

bool Iterator::HasNext() const {
  MX_DPTR(Iterator);
  return d != nullptr && d->HasNext();
}

RTValue Iterator::Next() const {
  MX_DPTR(Iterator);
  return d ? d->Next() : None;
}

RTValue Iterator::Next(bool* has_next) const {
  MX_DPTR(Iterator);
  return d ? d->Next(has_next) : None;
}

RTView Iterator::NextView(bool* has_next, RTValue* holder_or_null) const {
  MX_DPTR(Iterator);
  return d ? d->NextView(has_next, holder_or_null) : RTView();
}

int64_t Iterator::Distance() const {
  MX_DPTR(Iterator);
  return d ? d->Distance() : 0;
}

std::ostream& operator<<(std::ostream& os, Iterator const& n) {
  // TODO: GetRepr from Node
  os << "Iterator";
  return os;
}

}  // namespace runtime
}  // namespace matxscript
