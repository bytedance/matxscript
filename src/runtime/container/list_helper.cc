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
#include <matxscript/runtime/container/list_helper.h>
#include <matxscript/runtime/container/list_private.h>
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>
#include <matxscript/runtime/memory.h>
#include <unordered_set>

namespace matxscript {
namespace runtime {

bool is_comparable(const List& list) {
  int flag = 0;
  for (auto& item : list) {
    switch (item.type_code()) {
      case TypeIndex::kRuntimeInteger:
        flag |= 1;
        break;
      case TypeIndex::kRuntimeFloat:
        flag |= 2;
        break;
      case TypeIndex::kRuntimeString:
        flag |= 4;
        break;
      case TypeIndex::kRuntimeUnicode:
        flag |= 8;
        break;
      default:
        return false;
    }
  }
  if (flag < 4 || flag == 4 || flag == 8) {
    return true;
  }
  return false;
}

bool ListHelper::FirstShape(const List& list, std::vector<int64_t>& shape) {
  if (list.empty()) {
    return false;
  }
  shape.push_back(list.size());
  const RTValue& rt_value = list[0];
  if (rt_value.type_code() == TypeIndex::kRuntimeList) {
    ObjectView<List> sub_list_view = list[0].AsObjectViewNoCheck<List>();
    return FirstShape(sub_list_view.data(), shape);
  } else {
    return true;
  }
}

void ListHelper::Sort(const List& list) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    return;
  }
  /*
  if (!is_comparable(list)) {
    MXTHROW << "list_sort: only support List[number], List[str], List[bytes]";
  }
  */
  std::sort((p->data_container).begin(),
            (p->data_container).end(),
            [](const RTValue& x, const RTValue& y) { return !ArithOps::ge(x, y); });
}

void ListHelper::Sort(const List& list, const UserDataRef& comp) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    return;
  }
  std::sort((p->data_container).begin(),
            (p->data_container).end(),
            [&comp](const RTValue& x, const RTValue& y) -> bool {
              return comp.call(x, y).As<int64_t>() < 0;
            });
}

void ListHelper::NthElement(const List& list, int64_t n) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    return;
  }
  std::nth_element((p->data_container).begin(),
                   (p->data_container).begin() + n - 1,
                   (p->data_container).end(),
                   [](const RTValue& x, const RTValue& y) { return !ArithOps::ge(x, y); });
}

void ListHelper::NthElement(const List& list, int64_t n, const UserDataRef& comp) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    return;
  }
  std::nth_element((p->data_container).begin(),
                   (p->data_container).begin() + n - 1,
                   (p->data_container).end(),
                   [&comp](const RTValue& x, const RTValue& y) -> bool {
                     return comp.call(x, y).As<int64_t>() < 0;
                   });
}

void ListHelper::Heapify(const List& list) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    return;
  }
  std::make_heap((p->data_container).begin(),
                 (p->data_container).end(),
                 [](const RTValue& x, const RTValue& y) { return !ArithOps::ge(x, y); });
}

void ListHelper::Heapify(const List& list, const UserDataRef& comp) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    return;
  }
  std::sort((p->data_container).begin(),
            (p->data_container).end(),
            [&comp](const RTValue& x, const RTValue& y) -> bool {
              return comp.call(x, y).As<int64_t>() < 0;
            });
}

static void ShiftDown(std::vector<RTValue>& v,
                      size_t i,
                      const std::function<bool(const RTValue&, const RTValue&)>& func) {
  size_t min_index, left, right, length(v.size());
  while (true) {
    left = (i << 1) + 1;
    right = left + 1;
    min_index = i;
    if (left < length && func(v[left], v[min_index])) {
      min_index = left;
    }
    if (right < length && func(v[right], v[min_index])) {
      min_index = right;
    }
    if (min_index == i) {
      break;
    }
    std::swap(v[i], v[min_index]);
    i = min_index;
  }
}

void ListHelper::HeapReplace(const List& list, const Any& item) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    MXTHROW << "heap_replace: IndexError";
  }
  p->data_container[0] = item.As<RTValue>();
  ShiftDown(
      p->data_container, 0, [](const RTValue& x, const RTValue& y) { return !ArithOps::ge(x, y); });
}

void ListHelper::HeapReplace(const List& list, const Any& item, const UserDataRef& comp) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    MXTHROW << "heap_replace: IndexError";
  }
  p->data_container[0] = item.As<RTValue>();
  ShiftDown(p->data_container, 0, [&comp](const RTValue& x, const RTValue& y) -> bool {
    return comp.call(x, y).As<int64_t>() < 0;
  });
}

RTValue ListHelper::HeapPushPop(const List& list, const Any& item) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    MXTHROW << "heap_pushpop: IndexError";
  }
  if (ArithOps::ge((p->data_container)[0], item)) {
    return item.As<RTValue>();
  }
  RTValue ret = std::move((p->data_container)[0]);
  p->data_container[0] = item.As<RTValue>();
  ShiftDown(
      p->data_container, 0, [](const RTValue& x, const RTValue& y) { return !ArithOps::ge(x, y); });
  return ret;
}

RTValue ListHelper::HeapPushPop(const List& list, const Any& item, const UserDataRef& comp) {
  ListNode* p = list.GetListNode();
  if (list.size() == 0) {
    MXTHROW << "heap_pushpop: IndexError";
  }
  if (comp.call((p->data_container)[0], item).As<int64_t>() >= 0) {
    return item.As<RTValue>();
  }
  RTValue ret = std::move((p->data_container)[0]);
  p->data_container[0] = item.As<RTValue>();
  ShiftDown(p->data_container, 0, [&comp](const RTValue& x, const RTValue& y) -> bool {
    return comp.call(x, y).As<int64_t>() < 0;
  });
  return ret;
}

}  // namespace runtime
}  // namespace matxscript