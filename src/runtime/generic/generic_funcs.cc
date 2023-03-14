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
#include <matxscript/runtime/generic/generic_funcs.h>

#include <cstdint>
#include <cstdlib>

#include <matxscript/pipeline/pickle.h>
#include <matxscript/runtime/builtins_modules/_base64_util.h>
#include <matxscript/runtime/container/_list_helper.h>
#include <matxscript/runtime/container/list_helper.h>
#include <matxscript/runtime/container/ndarray_helper.h>
#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/unicode_view.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/env_time.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>
#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/jsonlib/json.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/stream_info.h>
#include <matxscript/runtime/utf8_util.h>
#ifndef DISABLE_UNICODEDATA
#include <matxscript/runtime/unicodelib/py_unicodedata.h>
#endif

namespace matxscript {
namespace runtime {

/******************************************************************************
 * user data custom method
 *****************************************************************************/
RTValue kernel_object___dispatch__(const Any& self, string_view func_name, PyArgs args) {
  MXCHECK(self.type_code() == TypeIndex::kRuntimeUserData)
      << self.type_name() << " has no method named " << func_name;
  return self.AsObjectViewNoCheck<UserDataRef>().data().generic_call_attr(func_name, args);
}

/******************************************************************************
 * python object data model special method names
 *****************************************************************************/

// Function signature is known
int64_t kernel_object___len__(const Any& self) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      return self.AsNoCheck<string_view>().size();
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return self.AsNoCheck<unicode_view>().size();
    } break;
    case TypeIndex::kRuntimeList: {
      return self.AsObjectViewNoCheck<List>().data().size();
    } break;
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectViewNoCheck<Set>().data().size();
    } break;
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectViewNoCheck<Dict>().data().size();
    } break;
    case TypeIndex::kRuntimeTuple: {
      return self.AsObjectViewNoCheck<Tuple>().data().size();
    } break;
    case TypeIndex::kRuntimeNDArray: {
      return self.AsObjectViewNoCheck<NDArray>().data().size();
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("__len__", {}).As<int64_t>();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("__len__", {}).As<int64_t>();
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"len\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return 0;
}

RTValue kernel_object___getitem__(const Any& self, const Any& key) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      int64_t index = key.As<int64_t>();
      return self.AsObjectViewNoCheck<List>().data().get_item(index);
    } break;
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectViewNoCheck<Dict>().data().get_item(key);
    } break;
    case TypeIndex::kRuntimeString: {
      int64_t index = key.As<int64_t>();
      return StringHelper::GetItem(self.AsNoCheck<string_view>(), index);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      int64_t index = key.As<int64_t>();
      return UnicodeHelper::GetItem(self.AsNoCheck<unicode_view>(), index);
    } break;
    case TypeIndex::kRuntimeTuple: {
      int64_t index = key.As<int64_t>();
      return self.AsObjectViewNoCheck<Tuple>().data().get_item(index);
    } break;
    case TypeIndex::kRuntimeNDArray: {
      return self.AsObjectViewNoCheck<NDArray>().data().get_item(key);
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("__getitem__", PyArgs(&key, 1));
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("__getitem__", PyArgs(&key, 1));
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"__getitem__\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return None;
}

RTValue kernel_object___setitem__(const Any& self, const Any& key, const Any& item) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      int64_t index = key.As<int64_t>();
      self.AsObjectViewNoCheck<List>().data().set_item(index, item.As<RTValue>());
    } break;
    case TypeIndex::kRuntimeDict: {
      self.AsObjectViewNoCheck<Dict>().data().set_item(key.As<RTValue>(), item.As<RTValue>());
    } break;
    case TypeIndex::kRuntimeNDArray: {
      self.AsObjectRefNoCheck<NDArray>().set_item(key, item);
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("__setitem__", {key.As<RTView>(), item.As<RTView>()});
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("__setitem__", {key.As<RTView>(), item.As<RTView>()});
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"__setitem__\"";
    } break;
  }
  return None;
}

RTValue kernel_object___delitem__(const Any& self, const Any& key) {
  MXTHROW << "\"" << self.type_name() << "\" object has no method \"__delitem__\"";
  return None;
}

RTValue kernel_object___getattr__(const Any& self, string_view attr) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().__getattr__(attr);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"__getattr__\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return None;
}

RTValue kernel_object___setattr__(const Any& self, string_view attr, const Any& item) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeUserData: {
      auto ud_ref = self.AsObjectRefNoCheck<UserDataRef>();
      ud_ref.set_attr(attr, item);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"__setattr__\"";
    } break;
  }
  return None;
}

RTValue kernel_object___getslice__(const Any& self,
                                   const Any& start,
                                   const Any& end,
                                   const Any& step) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      int64_t istart = start.As<int64_t>();
      int64_t iend = end.As<int64_t>();
      int64_t istep = step.As<int64_t>();
      return self.AsObjectViewNoCheck<List>().data().get_slice(istart, iend, istep);
    } break;
    case TypeIndex::kRuntimeTuple: {
      int64_t istart = start.As<int64_t>();
      int64_t iend = end.As<int64_t>();
      int64_t istep = step.As<int64_t>();
      return self.AsObjectViewNoCheck<Tuple>().data().get_slice(istart, iend, istep);
    } break;
    case TypeIndex::kRuntimeString: {
      int64_t istart = start.As<int64_t>();
      int64_t iend = end.As<int64_t>();
      int64_t istep = step.As<int64_t>();
      return StringHelper::GetSlice(self.AsNoCheck<string_view>(), istart, iend, istep);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      int64_t istart = start.As<int64_t>();
      int64_t iend = end.As<int64_t>();
      int64_t istep = step.As<int64_t>();
      return UnicodeHelper::GetSlice(self.AsNoCheck<unicode_view>(), istart, iend, istep);
    } break;
    case TypeIndex::kRuntimeNDArray: {
      int64_t istart = start.As<int64_t>();
      int64_t iend = end.As<int64_t>();
      int64_t istep = step.As<int64_t>();
      return self.AsObjectViewNoCheck<NDArray>().data().get_slice(istart, iend, istep);
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr(
          "__getslice__", {start.As<RTView>(), end.As<RTView>(), step.As<RTView>()});
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"__getslice__\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return None;
}

RTValue kernel_object___setslice__(const Any& self,
                                   const Any& start,
                                   const Any& end,
                                   const Any& item) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      int64_t istart = start.As<int64_t>();
      int64_t iend = end.As<int64_t>();
      auto rlist = item.As<List>();
      self.AsObjectViewNoCheck<List>().data().set_slice(istart, iend, rlist);
    } break;
    case TypeIndex::kRuntimeNDArray: {
      int64_t istart = start.As<int64_t>();
      int64_t iend = end.As<int64_t>();
      self.AsObjectRefNoCheck<NDArray>().set_slice(istart, iend, item);
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr(
          "__setslice__", {start.As<RTView>(), end.As<RTView>(), item.As<RTView>()});
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"__setslice__\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return None;
}

RTValue kernel_object___reversed__(const Any& self) {
  MXTHROW << "\"" << self.type_name() << "\" object has no method \"__reversed__\"";
  return None;
}

bool kernel_object___contains__(const Any& self, const Any& item) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      return StringHelper::Contains(self.AsNoCheck<string_view>(), item.As<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return UnicodeHelper::Contains(self.AsNoCheck<unicode_view>(), item.As<unicode_view>());
    } break;
    case TypeIndex::kRuntimeList: {
      return self.AsObjectViewNoCheck<List>().data().contains(item);
    } break;
    case TypeIndex::kRuntimeTuple: {
      return self.AsObjectViewNoCheck<Tuple>().data().contains(item);
    } break;
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectViewNoCheck<Set>().data().contains(item);
    } break;
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectViewNoCheck<Dict>().data().contains(item);
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("__contains__", PyArgs(&item, 1)).As<bool>();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("__contains__", PyArgs(&item, 1)).As<bool>();
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"__contains__\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return false;
}

RTValue kernel_object___hash__(const Any& self) {
  MXTHROW << "\"" << self.type_name() << "\" object has no method \"__hash__\"";
  return None;
}

/******************************************************************************
 * fused builtin object's special method
 *****************************************************************************/

// __fused_getitem__
RTValue kernel_object___fused_getitem__(const Any& self, const PyArgs& keys) {
  if (self.IsObjectRef<NDArray>()) {
    // optimize ndarray
    // keys must be int list
    std::vector<int64_t> int_keys;
    for (auto& k : keys) {
      int_keys.push_back(k.As<int64_t>());
    }
    return self.AsObjectViewNoCheck<NDArray>().data().fused_get_item(int_keys.data(),
                                                                     int_keys.size());
  } else {
    RTValue o = kernel_object___getitem__(self, keys[0]);
    for (int i = 1; i < keys.size(); ++i) {
      o = kernel_object___getitem__(o, keys[i]);
    }
    return o;
  }
}

// __fused_setitem__
RTValue kernel_object___fused_setitem__(const Any& self, const PyArgs& keys, const Any& item) {
  if (self.IsObjectRef<NDArray>()) {
    // optimize ndarray
    // keys must be int list
    std::vector<int64_t> int_keys;
    for (auto& k : keys) {
      int_keys.push_back(k.As<int64_t>());
    }
    self.AsObjectViewNoCheck<NDArray>().data().fused_set_item(
        int_keys.data(), int_keys.size(), item);
    return None;
  } else {
    if (keys.size() > 1) {
      RTValue o = kernel_object___getitem__(self, keys[0]);
      for (int i = 1; i < keys.size() - 1; ++i) {
        o = kernel_object___getitem__(o, keys[i]);
      }
      return kernel_object___setitem__(o, keys[keys.size() - 1], item);
    } else {
      return kernel_object___setitem__(self, keys[0], item);
    }
  }
}

/******************************************************************************
 * builtin object's member function
 *****************************************************************************/

// generic
RTValue kernel_object_append(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK_EQ(args.size(), 1) << "list.append Expect 1 arguments but get " << args.size();
      self.AsObjectViewNoCheck<List>().data().push_back(args[0].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("append", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("append", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"append\"";
    } break;
  }
  return None;
}

RTValue kernel_object_add(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      MXCHECK_EQ(args.size(), 1) << "set.add Expect 1 arguments but get " << args.size();
      self.AsObjectViewNoCheck<Set>().data().add(args[0].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("add", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("add", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"add\"";
    } break;
  }
  return None;
}

RTValue kernel_object_extend(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK_EQ(args.size(), 1) << "list.extend Expect 1 arguments but get " << args.size();
      MXCHECK(args[0].IsObjectRef<List>())
          << "\"" << args[0].type_name() << "\" is not a valid argument type. "
          << "You can only extend a List with List.";
      self.AsObjectViewNoCheck<List>().data().extend(args[0].AsObjectViewNoCheck<List>().data());
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("extend", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("extend", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"extend\"";
    } break;
  }
  return None;
}
RTValue kernel_object_clear(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK_EQ(args.size(), 0) << "list.clear Expect 0 arguments but get " << args.size();
      self.AsObjectViewNoCheck<List>().data().clear();
    } break;
    case TypeIndex::kRuntimeDict: {
      MXCHECK_EQ(args.size(), 0) << "dict.clear Expect 0 arguments but get " << args.size();
      self.AsObjectViewNoCheck<Dict>().data().clear();
    } break;
    case TypeIndex::kRuntimeSet: {
      MXCHECK_EQ(args.size(), 0) << "set.clear Expect 0 arguments but get " << args.size();
      self.AsObjectViewNoCheck<Set>().data().clear();
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("clear", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("clear", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"clear\"";
    } break;
  }
  return None;
}

RTValue kernel_object_reserve(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK_EQ(args.size(), 1) << "list.reserve Expect 1 arguments but get " << args.size();
      self.AsObjectViewNoCheck<List>().data().reserve(args[0].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeDict: {
      MXCHECK_EQ(args.size(), 1) << "dict.reserve Expect 1 arguments but get " << args.size();
      self.AsObjectViewNoCheck<Dict>().data().reserve(args[0].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeSet: {
      MXCHECK_EQ(args.size(), 1) << "set.reserve Expect 1 arguments but get " << args.size();
      self.AsObjectViewNoCheck<Set>().data().reserve(args[0].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("reserve", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("reserve", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"reserve\"";
    } break;
  }
  return None;
}

RTValue kernel_object_capacity(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK_EQ(args.size(), 0) << "list.capacity Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<List>().data().capacity();
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("capacity", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("capacity", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"capacity\"";
    } break;
  }

  // this is unreachable, just for disable warning!
  return 0;
}

RTValue kernel_object_bucket_count(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      MXCHECK_EQ(args.size(), 0) << "dict.bucket_count Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Dict>().data().bucket_count();
    } break;
    case TypeIndex::kRuntimeSet: {
      MXCHECK_EQ(args.size(), 0) << "set.bucket_count Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Set>().data().bucket_count();
    } break;
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("bucket_count", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("bucket_count", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"bucket_count\"";
    } break;
  }

  // this is unreachable, just for disable warning!
  return 0;
}

RTValue kernel_object_find(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeUnicode: {
      MXCHECK(args.size() >= 1 && args.size() <= 3)
          << "unicode.find Expect 1, 2 or 3 arguments but get " << args.size();
      if (args.size() == 1) {
        return UnicodeHelper::PyFind(self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>());
      } else if (args.size() == 2) {
        return UnicodeHelper::PyFind(
            self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>(), args[1].As<int64_t>());
      } else {
        return UnicodeHelper::PyFind(self.AsNoCheck<unicode_view>(),
                                     args[0].As<unicode_view>(),
                                     args[1].As<int64_t>(),
                                     args[2].As<int64_t>());
      }
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("find", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"find\"";
    } break;
  }
  return None;
}

RTValue kernel_object_update(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeTrie: {
      MXCHECK(args.size() == 1 || args.size() == 2)
          << "trie.update Expect 1 or 2 arguments but get " << args.size();
      if (args.size() == 1) {
        self.ptr<TrieNode>()->update(args[0]);
      } else {
        self.ptr<TrieNode>()->update(args[0], args[1].As<int64_t>());
      }
    } break;
    case TypeIndex::kRuntimeSet: {
      self.AsObjectViewNoCheck<Set>().data().update(args);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("update", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("update", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"update\"";
    } break;
  }
  return None;
}

RTValue container_min(const Iterator& iter) {
  auto iter_node = iter.GetMutableNode();
  MXCHECK(iter_node || iter_node->HasNext()) << "input is empty";
  RTValue min_value = iter_node->Next();
  while (iter_node->HasNext()) {
    RTValue item = iter_node->Next();
    if (ArithOps::lt(item, min_value)) {
      min_value = std::move(item);
    }
  }
  return min_value;
}

RTValue container_max(const Iterator& iter) {
  auto iter_node = iter.GetMutableNode();
  MXCHECK(iter_node || iter_node->HasNext()) << "input is empty";
  RTValue max_value = iter_node->Next();
  while (iter_node->HasNext()) {
    RTValue item = iter_node->Next();
    if (ArithOps::gt(item, max_value)) {
      max_value = std::move(item);
    }
  }
  return max_value;
}

// ndarray
NDArray kernel_nd_module_add(const Any& lhs, const Any& rhs) {
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Add(lhs.AsObjectViewNoCheck<NDArray>().data(),
                               rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeInteger) {
    return NDArrayOperate::Add(lhs.AsObjectViewNoCheck<NDArray>().data(), rhs.As<int64_t>());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeFloat) {
    return NDArrayOperate::Add(lhs.AsObjectViewNoCheck<NDArray>().data(), rhs.As<double>());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeInteger &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Add(rhs.AsObjectViewNoCheck<NDArray>().data(), lhs.As<int64_t>());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeFloat &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Add(rhs.AsObjectViewNoCheck<NDArray>().data(), lhs.As<double>());
  }
  MXTHROW << "NDArray add op only supports: "
          << "(NDArray,NDArray) and (NDArray, number)";
  return {};
}

NDArray kernel_nd_module_sub(const Any& lhs, const Any& rhs) {
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Sub(lhs.AsObjectViewNoCheck<NDArray>().data(),
                               rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeInteger) {
    return NDArrayOperate::Add(lhs.AsObjectViewNoCheck<NDArray>().data(), -(rhs.As<int64_t>()));
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeFloat) {
    return NDArrayOperate::Add(lhs.AsObjectViewNoCheck<NDArray>().data(), -(rhs.As<double>()));
  }
  if (lhs.type_code() == TypeIndex::kRuntimeInteger &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Sub(lhs.As<int64_t>(), rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeFloat &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Sub(lhs.As<double>(), rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  MXTHROW << "NDArray sub op only supports: "
          << "(NDArray,NDArray) and (NDArray, number)";
  return {};
}

NDArray kernel_nd_module_div(const Any& lhs, const Any& rhs) {
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Div(lhs.AsObjectViewNoCheck<NDArray>().data(),
                               rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeInteger) {
    return NDArrayOperate::Div(lhs.AsObjectViewNoCheck<NDArray>().data(),
                               (double)(rhs.As<int64_t>()));
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeFloat) {
    return NDArrayOperate::Div(lhs.AsObjectViewNoCheck<NDArray>().data(), rhs.As<double>());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeInteger &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Div((double)(lhs.As<int64_t>()),
                               rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeFloat &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Div(lhs.As<double>(), rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  MXTHROW << "NDArray div op only supports: "
          << "(NDArray,NDArray) and (NDArray, number)";
  return {};
}

NDArray kernel_nd_module_mul(const Any& lhs, const Any& rhs) {
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Mul(lhs.AsObjectViewNoCheck<NDArray>().data(),
                               rhs.AsObjectViewNoCheck<NDArray>().data());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeInteger) {
    return NDArrayOperate::Mul(lhs.AsObjectViewNoCheck<NDArray>().data(), rhs.As<int64_t>());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeNDArray &&
      rhs.type_code() == TypeIndex::kRuntimeFloat) {
    return NDArrayOperate::Mul(lhs.AsObjectViewNoCheck<NDArray>().data(), rhs.As<double>());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeInteger &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Mul(rhs.AsObjectViewNoCheck<NDArray>().data(), lhs.As<int64_t>());
  }
  if (lhs.type_code() == TypeIndex::kRuntimeFloat &&
      rhs.type_code() == TypeIndex::kRuntimeNDArray) {
    return NDArrayOperate::Mul(rhs.AsObjectViewNoCheck<NDArray>().data(), lhs.As<double>());
  }
  MXTHROW << "NDArray multiply op only supports: "
          << "(NDArray,NDArray) and (NDArray, number)";
  return {};
}

NDArray kernel_nd_module_rand(const Any& view) {
  MXCHECK(view.type_code() == TypeIndex::kRuntimeList) << "argument of matx.nd_rand must be List";
  const auto& obj_view = view.AsObjectViewNoCheck<List>();
  const List& obj = obj_view.data();
  std::vector<int64_t> shape(obj.size(), 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    MXCHECK(obj[i].type_code() == TypeIndex::kRuntimeInteger)
        << "matx.nd_rand: argument shape is invalid";
    shape[i] = obj[i].As<int64_t>();
    MXCHECK(shape[i] > 0) << "argument shape is invalid";
  }
  return NDArrayOperate::Rand(shape);
}

NDArray kernel_nd_module_concatenate(PyArgs args) {
  if (args.size() == 1) {
    return NDArrayOperate::Concatenate(args[0]);
  } else {
    return NDArrayOperate::Concatenate(args[0], args[1].As<int64_t>());
  }
}

NDArray kernel_nd_module_stack(PyArgs args) {
  if (args.size() == 1) {
    return NDArrayOperate::Stack(args[0]);
  } else {
    return NDArrayOperate::Stack(args[0], args[1].As<int64_t>());
  }
}

void kernel_list_module_sort(PyArgs args) {
  MXCHECK(args.size() == 1 || args.size() == 2)
      << "list_sort expect 1 or 2 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeList) << "list_sort: first arg must be List";
  if (args.size() == 1) {
    return ListHelper::Sort(args[0].AsObjectViewNoCheck<List>().data());
  } else {
    MXCHECK(args[1].type_code() == TypeIndex::kRuntimeUserData)
        << "list_sort: second arg must be UserDataRef";
    return ListHelper::Sort(args[0].AsObjectViewNoCheck<List>().data(),
                            args[1].AsObjectViewNoCheck<UserDataRef>().data());
  }
}

void kernel_list_module_nth_element(PyArgs args) {
  MXCHECK(args.size() == 2 || args.size() == 3)
      << "list_nth_element expect 2 or 3 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeList)
      << "list_nth_element: first arg must be List";
  if (args.size() == 2) {
    return ListHelper::NthElement(args[0].AsObjectViewNoCheck<List>().data(),
                                  args[1].As<int64_t>());
  } else {
    MXCHECK(args[2].type_code() == TypeIndex::kRuntimeUserData)
        << "list_nth_element: third arg must be UserDataRef";
    return ListHelper::NthElement(args[0].AsObjectViewNoCheck<List>().data(),
                                  args[1].As<int64_t>(),
                                  args[2].AsObjectViewNoCheck<UserDataRef>().data());
  }
}

void kernel_list_module_heapify(PyArgs args) {
  MXCHECK(args.size() == 1 || args.size() == 2)
      << "list_heapify expect 1 or 2 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeList) << "list_heapify: first arg must be List";
  if (args.size() == 1) {
    return ListHelper::Heapify(args[0].AsObjectViewNoCheck<List>().data());
  } else {
    MXCHECK(args[1].type_code() == TypeIndex::kRuntimeUserData)
        << "list_heapify: second arg must be UserDataRef";
    return ListHelper::Heapify(args[0].AsObjectViewNoCheck<List>().data(),
                               args[1].AsObjectViewNoCheck<UserDataRef>().data());
  }
}

void kernel_list_module_heap_replace(PyArgs args) {
  MXCHECK(args.size() == 2 || args.size() == 3)
      << "list_heap_replace expect 2 or 3 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeList)
      << "list_heap_replace: first arg must be List";
  if (args.size() == 2) {
    return ListHelper::HeapReplace(args[0].AsObjectViewNoCheck<List>().data(), args[1]);
  } else {
    MXCHECK(args[2].type_code() == TypeIndex::kRuntimeUserData)
        << "list_heap_replace: third arg must be UserDataRef";
    return ListHelper::HeapReplace(args[0].AsObjectViewNoCheck<List>().data(),
                                   args[1],
                                   args[2].AsObjectViewNoCheck<UserDataRef>().data());
  }
}

RTValue kernel_list_module_heap_pushpop(PyArgs args) {
  MXCHECK(args.size() == 2 || args.size() == 3)
      << "list_heap_pushpop expect 2 or 3 args, bug get " << args.size();
  MXCHECK(args[0].type_code() == TypeIndex::kRuntimeList)
      << "list_heap_pushpop: first arg must be List";
  if (args.size() == 2) {
    return ListHelper::HeapPushPop(args[0].AsObjectViewNoCheck<List>().data(), args[1]);
  } else {
    MXCHECK(args[2].type_code() == TypeIndex::kRuntimeUserData)
        << "list_heap_pushpop: third arg must be UserDataRef";
    return ListHelper::HeapPushPop(args[0].AsObjectViewNoCheck<List>().data(),
                                   args[1],
                                   args[2].AsObjectViewNoCheck<UserDataRef>().data());
  }
}

static void MATXCUDAStreamDeleter(void* self) {
  auto stream_info = reinterpret_cast<StreamInfo*>(self);
  MATXScriptStreamFree(DLDeviceType::kDLCUDA,
                       stream_info->device_id,
                       reinterpret_cast<MATXScriptStreamHandle>(stream_info->device_stream));

  stream_info->~StreamInfo();
}

OpaqueObject kernel_cuda_module_default_stream(int64_t device_id) {
  MXCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  MATXScriptStreamHandle stream = nullptr;
  OpaqueObject opaque_object = OpaqueObject();
  unsigned char* buffer_ptr = opaque_object.GetInternalBufferPtr();

  StreamInfo* stream_info = new (buffer_ptr)(StreamInfo);
  stream_info->device_id = device_id;
  stream_info->device_stream = stream;
  opaque_object.update(1, stream_info, MATXCUDAStreamDeleter);
  return opaque_object;
}

OpaqueObject kernel_cuda_module_create_stream(int64_t device_id) {
  MXCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  MATXScriptDevice device;
  device.device_id = device_id;
  device.device_type = DLDeviceType::kDLCUDA;
  MATXScriptStreamHandle stream = DeviceAPI::Get(device)->CreateStream(device);

  OpaqueObject opaque_object = OpaqueObject();
  unsigned char* buffer_ptr = opaque_object.GetInternalBufferPtr();

  StreamInfo* stream_info = new (buffer_ptr)(StreamInfo);
  stream_info->device_id = device_id;
  stream_info->device_stream = stream;
  opaque_object.update(1, stream_info, MATXCUDAStreamDeleter);
  return opaque_object;
}

void kernel_cuda_module_stream_sync(const OpaqueObject& stream, int64_t device_id) {
  if (device_id < 0) {
    THROW_PY_ValueError("stream_sync() Device Id must be equal or greater than zeros.");
  }
  StreamInfo* stream_info = reinterpret_cast<StreamInfo*>(stream.GetOpaquePtr());
  MATXScriptSynchronize(DLDeviceType::kDLCUDA,
                        stream_info->device_id,
                        reinterpret_cast<MATXScriptStreamHandle>(stream_info->device_stream));
}

void kernel_cuda_module_stream_sync(const Any& stream, int64_t device_id) {
  if (stream.type_code() == TypeIndex::kRuntimeOpaqueObject) {
    return kernel_cuda_module_stream_sync(stream.AsObjectViewNoCheck<OpaqueObject>().data(),
                                          device_id);
  } else {
    THROW_PY_TypeError("stream_sync() first arg must be OpaqueObject, not ", stream.type_name());
  }
}

template <typename T>
static auto _iter_min(T begin, T end) {
  T min_it = begin;
  T it = begin;
  ++it;
  for (; it != end; ++it) {
    if (ArithOps::lt(*it, *min_it)) {
      min_it = it;
    }
  }
  return *min_it;
}

template <typename T>
static auto _iter_max(T begin, T end) {
  T max_it = begin;
  T it = begin;
  ++it;
  for (; it != end; ++it) {
    if (ArithOps::gt(*it, *max_it)) {
      max_it = it;
    }
  }
  return *max_it;
}

RTValue kernel_math_iterable_min(const List& arg) {
  MXCHECK(!arg.empty()) << "input is empty";
  return _iter_min(arg.begin(), arg.end());
}

RTValue kernel_math_iterable_min(const Set& arg) {
  MXCHECK(!arg.empty()) << "input is empty";
  return _iter_min(arg.begin(), arg.end());
}

RTValue kernel_math_iterable_min(const Any& arg) {
  switch (arg.type_code()) {
    case TypeIndex::kRuntimeIterator: {
      return container_min(arg.AsObjectViewNoCheck<Iterator>().data());
    } break;
    case TypeIndex::kRuntimeSet: {
      return kernel_math_iterable_min(arg.AsObjectViewNoCheck<Set>().data());
    } break;
    case TypeIndex::kRuntimeList: {
      return kernel_math_iterable_min(arg.AsObjectViewNoCheck<List>().data());
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = arg.AsObjectViewNoCheck<FTObjectBase>();
      return kernel_math_iterable_min(
          ud_view.data().generic_call_attr("__iter__", {}).As<RTView>());
    } break;
    default: {
      return container_min(Kernel_Iterable::make(arg));
    } break;
  }
}

RTValue kernel_math_min(PyArgs args) {
  return _iter_min(args.begin(), args.end()).As<RTValue>();
}

RTValue kernel_math_iterable_max(const List& arg) {
  MXCHECK(!arg.empty()) << "input is empty";
  return _iter_max(arg.begin(), arg.end());
}

RTValue kernel_math_iterable_max(const Set& arg) {
  MXCHECK(!arg.empty()) << "input is empty";
  return _iter_max(arg.begin(), arg.end());
}

RTValue kernel_math_iterable_max(const Any& arg) {
  switch (arg.type_code()) {
    case TypeIndex::kRuntimeIterator: {
      return container_max(arg.AsObjectRefNoCheck<Iterator>());
    } break;
    case TypeIndex::kRuntimeSet: {
      return kernel_math_iterable_max(arg.AsObjectViewNoCheck<Set>().data());
    } break;
    case TypeIndex::kRuntimeList: {
      return kernel_math_iterable_max(arg.AsObjectViewNoCheck<List>().data());
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = arg.AsObjectViewNoCheck<FTObjectBase>();
      return kernel_math_iterable_max(
          ud_view.data().generic_call_attr("__iter__", {}).As<RTView>());
    } break;
    default: {
      return container_max(Kernel_Iterable::make(arg));
    } break;
  }
}

RTValue kernel_math_max(PyArgs args) {
  return _iter_max(args.begin(), args.end()).As<RTValue>();
}

// str/bytes/regex
RTValue kernel_object_lower(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      MXCHECK_EQ(args.size(), 0) << "bytes.lower Expect 0 arguments but get " << args.size();
      return StringHelper::Lower(self.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      MXCHECK_EQ(args.size(), 0) << "unicode.lower Expect 0 arguments but get " << args.size();
      return UnicodeHelper::Lower(self.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("lower", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"lower\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return None;
}
RTValue kernel_object_upper(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      MXCHECK_EQ(args.size(), 0) << "bytes.upper Expect 0 arguments but get " << args.size();
      return StringHelper::Upper(self.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      MXCHECK_EQ(args.size(), 0) << "unicode.upper Expect 0 arguments but get " << args.size();
      return UnicodeHelper::Upper(self.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("upper", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"upper\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return None;
}

RTValue kernel_object_isdigit(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      MXCHECK_EQ(args.size(), 0) << "bytes.isdigit Expect 0 arguments but get " << args.size();
      return StringHelper::Isdigit(self.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      MXCHECK_EQ(args.size(), 0) << "unicode.isdigit Expect 0 arguments but get " << args.size();
      return UnicodeHelper::IsDigit(self.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("isdigit", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"isdigit\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return false;
}

RTValue kernel_object_isalpha(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      MXCHECK_EQ(args.size(), 0) << "bytes.isalpha Expect 0 arguments but get " << args.size();
      return StringHelper::Isalpha(self.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      MXCHECK_EQ(args.size(), 0) << "unicode.isalpha Expect 0 arguments but get " << args.size();
      return UnicodeHelper::IsAlpha(self.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("isalpha", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"isalpha\"";
    } break;
  }
  // this is unreachable, just for disable warning!
  return false;
}

RTValue kernel_object_encode(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeUnicode: {
      MXCHECK_EQ(args.size(), 0) << "unicode.encode Expect 0 arguments but get " << args.size();
      return UnicodeHelper::Encode(self.AsNoCheck<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("encode", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"encode\"";
    } break;
  }
  return None;
}

RTValue kernel_object_decode(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      MXCHECK_EQ(args.size(), 0) << "bytes.decode Expect 0 arguments but get " << args.size();
      return StringHelper::Decode(self.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("decode", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"decode\"";
    } break;
  }
  return None;
}

RTValue kernel_object_split(const Any& self, PyArgs args) {
  auto num_args = args.size();
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      if (num_args > 0 && args[num_args - 1].type_code() == TypeIndex::kRuntimeKwargs) {
        static string_view arg_names[2] = {"sep", "maxsplit"};
        static RTView default_args[2] = {None, RTView(-1)};
        KwargsUnpackHelper helper("bytes.split", arg_names, 2, default_args, 2);
        RTView pos_args[2];
        helper.unpack(pos_args, args);
        return StringHelper::Split(self.AsNoCheck<string_view>(),
                                   pos_args[0].As<string_view>(),
                                   pos_args[1].As<int64_t>());
      } else {
        switch (num_args) {
          case 0: {
            return StringHelper::Split(self.AsNoCheck<string_view>());
          } break;
          case 1: {
            return StringHelper::Split(self.AsNoCheck<string_view>(), args[0].As<string_view>());
          } break;
          case 2: {
            return StringHelper::Split(
                self.AsNoCheck<string_view>(), args[0].As<string_view>(), args[1].As<int64_t>());
          } break;
          default: {
            THROW_PY_TypeError("split() takes at most 2 arguments (", num_args, " given)");
          } break;
        }
      }
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (num_args > 0 && args[num_args - 1].type_code() == TypeIndex::kRuntimeKwargs) {
        static string_view arg_names[2] = {"sep", "maxsplit"};
        static RTView default_args[2] = {None, RTView(-1)};
        KwargsUnpackHelper helper("str.split", arg_names, 2, default_args, 2);
        RTView pos_args[2];
        helper.unpack(pos_args, args);
        return UnicodeHelper::Split(self.AsNoCheck<unicode_view>(),
                                    pos_args[0].As<unicode_view>(),
                                    pos_args[1].As<int64_t>());
      } else {
        switch (num_args) {
          case 0: {
            return UnicodeHelper::Split(self.AsNoCheck<unicode_view>());
          } break;
          case 1: {
            return UnicodeHelper::Split(self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>());
          } break;
          case 2: {
            return UnicodeHelper::Split(
                self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>(), args[1].As<int64_t>());
          } break;
          default: {
            THROW_PY_TypeError("split() takes at most 2 arguments (", num_args, " given)");
          } break;
        }
      }
    } break;
#ifdef MATX_ENABLE_PCRE_REGEX
    case TypeIndex::kRuntimeRegex: {
      MXCHECK(args.size() == 1) << "re.split Expect 1 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Regex>().data().split(args[0]);
    } break;
#endif
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("split", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"split\"";
    } break;
  }
  return None;
}

RTValue kernel_object_join(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      MXCHECK(args.size() == 1) << "bytes.join Expect 1 arguments but get " << args.size();
      return StringHelper::Join(self.AsNoCheck<string_view>(), args[0]);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      MXCHECK(args.size() == 1) << "unicode.join Expect 1 arguments but get " << args.size();
      return UnicodeHelper::Join(self.AsNoCheck<unicode_view>(), args[0]);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("join", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"join\"";
    } break;
  }
  return None;
}

RTValue kernel_object_replace(const Any& self, PyArgs args) {
  switch (self.type_code()) {
#ifdef MATX_ENABLE_PCRE_REGEX
    case TypeIndex::kRuntimeRegex: {
      MXCHECK(args.size() == 2) << "re.replace Expect 2 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Regex>().data().replace(args[0], args[1]);
    } break;
#endif
    case TypeIndex::kRuntimeString: {
      if (args.size() == 2) {
        return StringHelper::Replace(
            self.AsNoCheck<string_view>(), args[0].As<string_view>(), args[1].As<string_view>());
      }
      MXCHECK(args.size() == 3) << "bytes.replace Expect 2 or 3 arguments but get " << args.size();
      return StringHelper::Replace(self.AsNoCheck<string_view>(),
                                   args[0].As<string_view>(),
                                   args[1].As<string_view>(),
                                   args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (args.size() == 2) {
        return UnicodeHelper::Replace(
            self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>(), args[1].As<unicode_view>());
      }
      MXCHECK(args.size() == 3) << "unicode.replace Expect 2 or 3 arguments but get "
                                << args.size();
      return UnicodeHelper::Replace(self.AsNoCheck<unicode_view>(),
                                    args[0].As<unicode_view>(),
                                    args[1].As<unicode_view>(),
                                    args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("replace", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"split\"";
    } break;
  }
  return None;
}
RTValue kernel_object_match(const Any& self, PyArgs args) {
  switch (self.type_code()) {
#ifdef MATX_ENABLE_PCRE_REGEX
    case TypeIndex::kRuntimeRegex: {
      int64_t offset = 0;
      if (args.size() == 2) {
        offset = args[1].As<int64_t>();
      }
      return self.AsObjectViewNoCheck<Regex>().data().match(args[0], offset);
    } break;
#endif
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("match", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"match\"";
    } break;
  }
  return None;
}

RTValue kernel_object_startswith(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      if (args.size() == 1) {
        return StringHelper::StartsWith(self.AsNoCheck<string_view>(), args[0]);
      } else if (args.size() == 2) {
        return StringHelper::StartsWith(
            self.AsNoCheck<string_view>(), args[0], args[1].As<int64_t>());
      }
      MXCHECK(args.size() == 3) << "bytes.startswith Expect 1, 2 or 3 arguments but get "
                                << args.size();
      return StringHelper::StartsWith(
          self.AsNoCheck<string_view>(), args[0], args[1].As<int64_t>(), args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (args.size() == 1) {
        return UnicodeHelper::StartsWith(self.AsNoCheck<unicode_view>(), args[0]);
      } else if (args.size() == 2) {
        return UnicodeHelper::StartsWith(
            self.AsNoCheck<unicode_view>(), args[0], args[1].As<int64_t>());
      }
      MXCHECK(args.size() == 3) << "unicode.startswith Expect 1, 2 or 3 arguments but get "
                                << args.size();
      return UnicodeHelper::StartsWith(
          self.AsNoCheck<unicode_view>(), args[0], args[1].As<int64_t>(), args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("startswith", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" container has no method \"startswith\"";
    } break;
  }
  return None;
}

RTValue kernel_object_endswith(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      if (args.size() == 1) {
        return StringHelper::EndsWith(self.AsNoCheck<string_view>(), args[0]);
      } else if (args.size() == 2) {
        return StringHelper::EndsWith(
            self.AsNoCheck<string_view>(), args[0], args[1].As<int64_t>());
      }
      MXCHECK(args.size() == 3) << "bytes.endswith Expect 1, 2 or 3 arguments but get "
                                << args.size();
      return StringHelper::EndsWith(
          self.AsNoCheck<string_view>(), args[0], args[1].As<int64_t>(), args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (args.size() == 1) {
        return UnicodeHelper::EndsWith(self.AsNoCheck<unicode_view>(), args[0]);
      } else if (args.size() == 2) {
        return UnicodeHelper::EndsWith(
            self.AsNoCheck<unicode_view>(), args[0], args[1].As<int64_t>());
      }
      MXCHECK(args.size() == 3) << "unicode.endswith Expect 1, 2 or 3 arguments but get "
                                << args.size();
      return UnicodeHelper::EndsWith(
          self.AsNoCheck<unicode_view>(), args[0], args[1].As<int64_t>(), args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("endswith", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" container has no method \"endswith\"";
    } break;
  }
  return None;
}

RTValue kernel_object_lstrip(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      if (args.size() == 0) {
        return StringHelper::LStrip(self.AsNoCheck<string_view>());
      }
      MXCHECK(args.size() == 1) << "bytes.lstrip Expect 0 or 1 argument but get " << args.size();
      return StringHelper::LStrip(self.AsNoCheck<string_view>(), args[0].As<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (args.size() == 0) {
        return UnicodeHelper::LStrip(self.AsNoCheck<unicode_view>());
      }
      MXCHECK(args.size() == 1) << "unicode.lstrip Expect 0 or 1 argument but get " << args.size();
      return UnicodeHelper::LStrip(self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("lstrip", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" container has no method \"lstrip\"";
    } break;
  }
  return None;
}

RTValue kernel_object_rstrip(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      if (args.size() == 0) {
        return StringHelper::RStrip(self.AsNoCheck<string_view>());
      }
      MXCHECK(args.size() == 1) << "bytes.rstrip Expect 0 or 1 argument but get " << args.size();
      return StringHelper::RStrip(self.AsNoCheck<string_view>(), args[0].As<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (args.size() == 0) {
        return UnicodeHelper::RStrip(self.AsNoCheck<unicode_view>());
      }
      MXCHECK(args.size() == 1) << "unicode.rstrip Expect 0 or 1 argument but get " << args.size();
      return UnicodeHelper::RStrip(self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("rstrip", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" container has no method \"rstrip\"";
    } break;
  }
  return None;
}

RTValue kernel_object_strip(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      if (args.size() == 0) {
        return StringHelper::Strip(self.AsNoCheck<string_view>());
      }
      MXCHECK(args.size() == 1) << "bytes.strip Expect 0 or 1 argument but get " << args.size();
      return StringHelper::Strip(self.AsNoCheck<string_view>(), args[0].As<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (args.size() == 0) {
        return UnicodeHelper::Strip(self.AsNoCheck<unicode_view>());
      }
      MXCHECK(args.size() == 1) << "unicode.strip Expect 0 or 1 argument but get " << args.size();
      return UnicodeHelper::Strip(self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("strip", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" container has no method \"strip\"";
    } break;
  }
  return None;
}

RTValue kernel_object_count(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeString: {
      if (args.size() == 1) {
        return StringHelper::Count(self.AsNoCheck<string_view>(), args[0].As<string_view>());
      } else if (args.size() == 2) {
        return StringHelper::Count(
            self.AsNoCheck<string_view>(), args[0].As<string_view>(), args[1].As<int64_t>());
      }
      MXCHECK(args.size() == 3) << "bytes.count Expect 1, 2 or 3 arguments but get " << args.size();
      return StringHelper::Count(self.AsNoCheck<string_view>(),
                                 args[0].As<string_view>(),
                                 args[1].As<int64_t>(),
                                 args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      if (args.size() == 1) {
        return UnicodeHelper::Count(self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>());
      } else if (args.size() == 2) {
        return UnicodeHelper::Count(
            self.AsNoCheck<unicode_view>(), args[0].As<unicode_view>(), args[1].As<int64_t>());
      }
      MXCHECK(args.size() == 3) << "unicode.count Expect 1, 2 or 3 arguments but get "
                                << args.size();
      return UnicodeHelper::Count(self.AsNoCheck<unicode_view>(),
                                  args[0].As<unicode_view>(),
                                  args[1].As<int64_t>(),
                                  args[2].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeList: {
      MXCHECK(args.size() == 1) << "list.count Expect 1 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<List>().data().count(args[0]);
    } break;
    case TypeIndex::kRuntimeTuple: {
      MXCHECK(args.size() == 1) << "tuple.count Expect 1 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Tuple>().data().count(args[0]);
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("count", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("count", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" container has no method \"count\"";
    } break;
  }
  return None;
}

RTValue kernel_object_format(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeUnicode: {
      return UnicodeHelper::Format(self.As<unicode_view>(), args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("format", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" container has no method \"format\"";
    } break;
  }
  return None;
}

// dict
RTValue kernel_object_keys(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      MXCHECK_EQ(args.size(), 0) << "dict.keys Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Dict>().data().key_iter();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("keys", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("keys", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"keys\"";
    } break;
  }
  return None;
}

RTValue kernel_object_values(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      MXCHECK_EQ(args.size(), 0) << "dict.values Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Dict>().data().value_iter();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("values", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("values", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"values\"";
    } break;
  }
  return None;
}

RTValue kernel_object_items(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      MXCHECK_EQ(args.size(), 0) << "dict.items Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Dict>().data().item_iter();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("items", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("items", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"items\"";
    } break;
  }
  return None;
}

RTValue kernel_object_get(const Any& self, PyArgs args) {
  auto args_num = args.size();
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      MXCHECK(args_num == 1 || args_num == 2)
          << "dict.get Expect 1 or 2 arguments but get" << args.size();
      if (args.size() == 1) {
        return self.AsObjectViewNoCheck<Dict>().data().get_default(args[0].As<RTValue>(), None);
      } else {
        return self.AsObjectViewNoCheck<Dict>().data().get_default(args[0].As<RTValue>(),
                                                                   args[1].As<RTValue>());
      }
    } break;
    case TypeIndex::kRuntimeFTDict: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("get", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("get", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"get\"";
    } break;
  }
  return None;
}

// set
RTValue kernel_object_union(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectViewNoCheck<Set>().data().set_union(args);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("union", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("union", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"union\"";
    } break;
  }
  return None;
}

RTValue kernel_object_difference(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectViewNoCheck<Set>().data().difference(args);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("difference", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("difference", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"difference\"";
    } break;
  }
  return None;
}

RTValue kernel_object_difference_update(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      self.AsObjectViewNoCheck<Set>().data().difference_update(args);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("difference_update", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("difference_update", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"difference_update\"";
    } break;
  }
  return None;
}

RTValue kernel_object_discard(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      MXCHECK_EQ(args.size(), 1) << "set.discard Expect 1 arguments but get " << args.size();
      self.AsObjectViewNoCheck<Set>().data().discard(args[0]);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("discard", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("discard", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"discard\"";
    } break;
  }
  return None;
}

// NDArray
RTValue kernel_object_to_list(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 0) << "ndarray.to_list Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().ToList();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("to_list", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"to_list\"";
    } break;
  }
  return None;
}

RTValue kernel_object_tolist(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 0) << "ndarray.tolist Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().ToList();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("tolist", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"tolist\"";
    } break;
  }
  return None;
}

RTValue kernel_object_is_contiguous(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 0) << "ndarray.is_contiguous Expect 0 arguments but get "
                                 << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().IsContiguous();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("is_contiguous", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"is_contiguous\"";
    } break;
  }
  return None;
}

RTValue kernel_object_contiguous(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 0) << "ndarray.contiguous Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().Contiguous();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("contiguous", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"contiguous\"";
    } break;
  }
  return None;
}

RTValue kernel_object_reshape(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 1) << "ndarray.reshape Expect 1 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().Reshape(args[0]);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("reshape", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"reshape\"";
    } break;
  }
  return None;
}

RTValue kernel_object_squeeze(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 1) << "ndarray.squeeze Expect 1 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().Squeeze(
          args[0].AsObjectRefNoCheck<Tuple>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("squeeze", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"squeeze\"";
    } break;
  }
  return None;
}

RTValue kernel_object_unsqueeze(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 1) << "ndarray.unsqueeze Expect 1 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().Unsqueeze(args[0].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("squeeze", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"squeeze\"";
    } break;
  }
  return None;
}

RTValue kernel_object_shape(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 0) << "ndarray.shape Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().ShapeList();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("shape", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"shape\"";
    } break;
  }
  return None;
}

RTValue kernel_object_dtype(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 0) << "ndarray.dtype Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().DTypeUnicode();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("dtype", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"dtype\"";
    } break;
  }
  return None;
}

RTValue kernel_object_dim(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      if (args.size() != 0) {
        THROW_PY_TypeError("ndarray.dim() takes no arguments (", args.size(), " given)");
      }
      return self.AsObjectViewNoCheck<NDArray>().data().GetDim();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("dim", args);
    } break;
    default: {
      THROW_PY_AttributeError("'", self.type_name(), "' object has no attribute 'dim'");
    } break;
  }
  return None;
}

RTValue kernel_object_device(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK_EQ(args.size(), 0) << "ndarray.device Expect 0 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().Device();
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("device", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"device\"";
    } break;
  }
  return None;
}

RTValue kernel_object_transpose(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK(args.size() == 0 || args.size() == 1)
          << "ndarray.transpose Expect 0 or 1 arguments, but get " << args.size();
      if (args.size() == 0) {
        return self.AsObjectViewNoCheck<NDArray>().data().transpose();
      } else {
        return self.AsObjectViewNoCheck<NDArray>().data().transpose(args[0]);
      }
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("transpose", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"transpose\"";
    } break;
  }
  return None;
}

RTValue kernel_object_as_type(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeNDArray: {
      MXCHECK(args.size() == 1) << "ndarray.as_type Expect 1 arguments, but get " << args.size();
      return self.AsObjectViewNoCheck<NDArray>().data().as_type(args[0].As<Unicode>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("as_type", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"as_type\"";
    } break;
  }
  return None;
}

// trie tree
RTValue kernel_object_prefix_search(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeTrie: {
      MXCHECK(args.size() == 1 || args.size() == 2)
          << "trie.prefix_search Expect 1 or 2 arguments but get " << args.size();
      int64_t pos = 0;
      if (args.size() == 2) {
        pos = args[1].As<int64_t>();
      }
      return self.ptr<TrieNode>()->prefix_search(args[0], pos);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("prefix_search", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"prefix_search\"";
    } break;
  }
  return None;
}

RTValue kernel_object_prefix_search_all(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeTrie: {
      MXCHECK(args.size() == 1 || args.size() == 2)
          << "trie.prefix_search_all Expect 1 or 2 arguments but get " << args.size();
      int64_t pos = 0;
      if (args.size() == 2) {
        pos = args[1].As<int64_t>();
      }
      return self.ptr<TrieNode>()->prefix_search_all(args[0], pos);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("prefix_search", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"prefix_search\"";
    } break;
  }
  return None;
}

RTValue kernel_object_save(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeTrie: {
      MXCHECK(args.size() == 1) << "trie.save Expect 1 arguments but get " << args.size();
      return self.ptr<TrieNode>()->save(args[0].As<Unicode>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("save", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"save\"";
    } break;
  }
  return None;
}

RTValue kernel_object_load(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeTrie: {
      MXCHECK(args.size() == 1) << "trie.load Expect 1 arguments but get " << args.size();
      return self.ptr<TrieNode>()->load(args[0].As<Unicode>());
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("load", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"load\"";
    } break;
  }
  return None;
}

RTValue kernel_object_read(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeFile: {
      if (args.size() > 1) {
        THROW_PY_TypeError("read expected at most 1 argument, got ", args.size());
      }
      if (args.size() == 0) {
        return self.ptr<FileNode>()->Read();
      } else {
        if (args[0].IsObjectRef<Kwargs>()) {
          THROW_PY_TypeError("read() takes no keyword arguments");
        }
        if (args[0].is_nullptr()) {
          return self.ptr<FileNode>()->Read();
        } else if (args[0].Is<int64_t>()) {
          return self.ptr<FileNode>()->Read(args[0].AsNoCheck<int64_t>());
        } else {
          THROW_PY_TypeError("argument should be integer or None, not '", args[0].type_name(), "'");
        }
      }
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"read\"";
    } break;
  }
  return None;
}

/******************************************************************************
 * python simple builtin modules and functions
 *****************************************************************************/
#ifndef DISABLE_UNICODEDATA
// unicodedata
Unicode kernel_unicodedata_normalize(int32_t form, const unicode_view& s) {
  PyUnicodeData ud;
  return ud.normalize(form, s);
}

Unicode kernel_unicodedata_normalize(const unicode_view& form_s, const unicode_view& s) {
  int32_t form = 0;
  if (form_s == U"NFC") {
    form = 0;
  } else if (form_s == U"NFKC") {
    form = 1;
  } else if (form_s == U"NFD") {
    form = 2;
  } else if (form_s == U"NFKD") {
    form = 3;
  } else {
    THROW_PY_ValueError("invalid normalization form");
  }
  PyUnicodeData ud;
  return ud.normalize(form, s);
}

Unicode kernel_unicodedata_category(const unicode_view& s) {
  PyUnicodeData ud;
  if (s.size() != 1) {
    THROW_PY_TypeError("category() argument must be a unicode character, not str");
  }
  return StringHelper::Decode(ud.category(s[0]));
}
#endif

// ord
int64_t kernel_builtins_ord(const string_view& c) {
  if (c.size() != 1) {
    THROW_PY_TypeError("ord() expected a character, but string of length ", c.size(), " found");
  }
  return static_cast<int64_t>(c.data()[0]);
}
int64_t kernel_builtins_ord(const unicode_view& c) {
  if (c.size() != 1) {
    THROW_PY_TypeError("ord() expected a character, but string of length ", c.size(), " found");
  }
  return static_cast<int64_t>(c[0]);
}
int64_t kernel_builtins_ord(const Any& c) {
  switch (c.type_code()) {
    case TypeIndex::kRuntimeString: {
      return kernel_builtins_ord(c.AsNoCheck<string_view>());
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return kernel_builtins_ord(c.AsNoCheck<unicode_view>());
    } break;
    default: {
      THROW_PY_TypeError("ord() expected string of length 1, but ", c.type_name(), " found");
    } break;
  }
  return 0;
}

// chr
Unicode kernel_builtins_chr(int64_t i) {
  if (i < 0 || i >= 0x110000) {
    THROW_PY_ValueError("chr() arg not in range(0x110000)");
  }
  Unicode::value_type c = i;
  return Unicode(1, c);
}

// json
RTValue kernel_json_load(PyArgs args) {
  MXCHECK(args.size() == 1) << "json.load Expect 1 arguments but get " << args.size();
  auto fp_view = args[0].AsObjectView<File>();
  return json_load(fp_view.data());
}

RTValue kernel_json_loads(PyArgs args) {
  MXCHECK(args.size() == 1) << "json.loads Expect 1 arguments but get " << args.size();
  if (args[0].type_code() == TypeIndex::kRuntimeUnicode) {
    return json_loads(UTF8Encode(args[0].AsNoCheck<unicode_view>()));
  }
  return json_loads(args[0].As<string_view>());
}

// RTValue kernel_json_dump(PyArgs args);
Unicode kernel_json_dumps(PyArgs args) {
  if (args.size() == 1) {
    return json_dumps(args[0]);
  }
  if (args.size() == 2) {
    return json_dumps(args[0], args[1].As<int64_t>());
  }
  MXCHECK(args.size() == 3) << "json.loads Expect 1-3 arguments but get " << args.size();
  return json_dumps(args[0], args[1].As<int64_t>(), args[1].As<int64_t>());
}

// file
File kernel_file_open(PyArgs args) {
  MXCHECK(args.size() >= 1 && args.size() <= 3) << "Expect 1-3 arguments but get " << args.size();
  auto path = Unicode(args[0].As<unicode_view>());
  if (args.size() == 1) {
    return File(path);
  }
  auto mode = Unicode(args[1].As<unicode_view>());
  if (args.size() == 2) {
    return File(path, mode);
  }
  auto encoding = Unicode(args[2].As<unicode_view>());
  return File(path, mode, encoding);
}

void kernel_file_close(const File& f) {
  return f.close();
}

// list
RTValue kernel_object_pop(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK(args.size() == 0 || args.size() == 1)
          << "list.pop Expect 0 or 1 arguments but get " << args.size();
      if (args.size() == 0) {
        return self.AsObjectViewNoCheck<List>().data().pop();
      } else {
        return self.AsObjectViewNoCheck<List>().data().pop(args[0].As<int64_t>());
      }
    } break;
    case TypeIndex::kRuntimeDict: {
      MXCHECK(args.size() == 1 || args.size() == 2)
          << "dict.pop Expect 1 or 2 arguments but get " << args.size();
      return self.AsObjectViewNoCheck<Dict>().data().pop(args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("pop", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"pop\"";
    } break;
  }
  return None;
}

RTValue kernel_object_insert(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK(args.size() == 2) << "list.insert Expect 2 arguments but get " << args.size();
      self.AsObjectViewNoCheck<List>().data().insert(args[0].As<int64_t>(), args[1].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("insert", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"insert\"";
    } break;
  }
  return None;
}

RTValue kernel_object_remove(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK_EQ(args.size(), 1) << "list.remove Expect 1 arguments but get " << args.size();
      self.AsObjectViewNoCheck<List>().data().remove(args[0].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("remove", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("remove", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"remove\"";
    } break;
  }
  return None;
}

RTValue kernel_object_reverse(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      MXCHECK_EQ(args.size(), 0) << "list.remove Expect 0 arguments but get " << args.size();
      self.AsObjectViewNoCheck<List>().data().reverse();
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("reverse", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("reverse", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"reverse\"";
    } break;
  }
  return None;
}

RTValue kernel_object_sort(const Any& self, PyArgs args) {
  auto num_args = args.size();
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      if (num_args > 0) {
        RTValue* key_func = nullptr;
        bool reverse = false;
        list_details::trait_sort_kwargs(args, &key_func, &reverse);
        if (key_func) {
          self.AsObjectViewNoCheck<List>().data().sort(*key_func, reverse);
        } else {
          self.AsObjectViewNoCheck<List>().data().sort(reverse);
        }
      } else {
        self.AsObjectViewNoCheck<List>().data().sort();
      }
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("sort", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("sort", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"sort\"";
    } break;
  }
  return None;
}

RTValue kernel_object_index(const Any& self, PyArgs args) {
  auto num_args = args.size();
  switch (self.type_code()) {
    case TypeIndex::kRuntimeList: {
      switch (num_args) {
        case 1: {
          auto list_view = self.AsObjectViewNoCheck<List>();
          RTValue x = args[0].As<RTValue>();
          int64_t start = 0;
          int64_t end = list_view.data().size();
          return list_view.data().index(std::move(x), start, end);
        } break;
        case 2: {
          auto list_view = self.AsObjectViewNoCheck<List>();
          RTValue x = args[0].As<RTValue>();
          int64_t start = args[1].As<int64_t>();
          int64_t end = list_view.data().size();
          return list_view.data().index(std::move(x), start, end);
        } break;
        case 3: {
          auto list_view = self.AsObjectViewNoCheck<List>();
          RTValue x = args[0].As<RTValue>();
          int64_t start = args[1].As<int64_t>();
          int64_t end = args[2].As<int64_t>();
          return list_view.data().index(std::move(x), start, end);
        } break;
        default: {
          MXTHROW << "TypeError: index expected at most 3 arguments, got 4";
        } break;
      }
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = self.AsObjectViewNoCheck<FTObjectBase>();
      return ud_view.data().generic_call_attr("index", args);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = self.AsObjectViewNoCheck<UserDataRef>();
      return ud_view.data().generic_call_attr("index", args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object has no method \"sort\"";
    } break;
  }
  return None;
}

// time
double kernel_time_time() {
  double timestamp = EnvTime::Default()->NowNanos() * 1.0 / EnvTime::kSecondsToNanos;
  return timestamp;
}

// os
RTValue kernel_os_getenv(PyArgs args) {
  MXCHECK(args.size() == 1 || args.size() == 2)
      << "os.getenv Expect 1 or 2 arguments but get " << args.size();
  unicode_view key = args[0].As<unicode_view>();
  const char* env_p = std::getenv(UTF8Encode(key).c_str());
  if (env_p != nullptr) {
    return String(env_p).decode();
  } else {
    if (args.size() == 1) {
      return None;
    } else {
      return args[1].As<RTValue>();
    }
  }
}

// base64
String kernel_base64_b64encode(string_view s, RTView altchars) {
  // altchars is not supported now
  return py_builtins::base64_encode(s);
}

String kernel_base64_b64decode(string_view s, RTView altchars, bool validate) {
  // altchars and validate are not supported now
  return py_builtins::base64_decode(s);
}

// sorted
List kernel_builtins_sorted(const List& iterable, const Any& key, bool reverse) {
  auto new_list = iterable;
  if (key.is_nullptr()) {
    new_list.sort(reverse);
  } else {
    new_list.sort(key, reverse);
  }
  return new_list;
}

List kernel_builtins_sorted(const Tuple& iterable, const Any& key, bool reverse) {
  List new_list(iterable.begin(), iterable.end());
  if (key.is_nullptr()) {
    new_list.sort(reverse);
  } else {
    new_list.sort(key, reverse);
  }
  return new_list;
}

RTValue kernel_builtins_sorted(const Any& iterable, const Any& key, bool reverse) {
  switch (iterable.type_code()) {
    case TypeIndex::kRuntimeFTList: {
      auto ud_view = iterable.AsObjectViewNoCheck<FTObjectBase>();
      Iterator iter = ud_view.data().generic_call_attr("__iter__", {}).As<Iterator>();
      List new_list;
      while (iter.HasNext()) {
        new_list.push_back(iter.Next());
      }
      if (key.is_nullptr()) {
        new_list.sort(reverse);
      } else {
        new_list.sort(key, reverse);
      }
      return new_list;
    };
    case TypeIndex::kRuntimeUserData: {
      auto ud_view = iterable.AsObjectViewNoCheck<UserDataRef>();
      Iterator iter = ud_view.data().generic_call_attr("__iter__", {}).As<Iterator>();
      List new_list;
      while (iter.HasNext()) {
        new_list.push_back(iter.Next());
      }
      if (key.is_nullptr()) {
        new_list.sort(reverse);
      } else {
        new_list.sort(key, reverse);
      }
      return new_list;
    };
    default: {
      MXTHROW << "\"" << iterable.type_name() << "\" object does not support \"sorted\"";
      return List();
    };
  }
}

Iterator kernel_builtins_iter(const List& iterable) {
  return List::builtins_iter(iterable);
}

Iterator kernel_builtins_reversed(const List& iterable) {
  return List::builtins_reversed(iterable);
}

/******************************************************************************
 * python builtin modules and functions
 *
 * Function schema:
 *     RTValue module_method(self, *args);
 *
 *****************************************************************************/

RTValue kernel_builtins_print(PyArgs args, string_view sep, string_view end, FILE* file) {
  std::stringstream ss;
  for (auto i = 0; i < args.size(); ++i) {
    if (i > 0) {
      ss << sep;
    }
    if (args[i].IsString()) {
      ss << "b'" << args[i].AsNoCheck<string_view>() << "'";
    } else if (args[i].IsUnicode()) {
      ss << args[i].AsNoCheck<Unicode>();
    } else {
      ss << args[i];
    }
  }
  ss << end;
  auto repr = ss.str();
  fprintf(file, "%*s", (int)repr.size(), repr.c_str());
  return None;
}

// UserData object call.
RTValue kernel_object_call(const Any& self, PyArgs args) {
  switch (self.type_code()) {
    case TypeIndex::kRuntimeUserData: {
      return self.AsObjectViewNoCheck<UserDataRef>().data().generic_call(args);
    } break;
    default: {
      MXTHROW << "\"" << self.type_name() << "\" object should be a callable object.";
    } break;
  }
  return None;
}

Unicode kernel_pickle_serialize(const Any& o) {
  return pickle::Serialize(o).decode();
}

RTValue kernel_pickle_deserialize(unicode_view s) {
  return pickle::DeSerialize(UnicodeHelper::Encode(s));
}

}  // namespace runtime
}  // namespace matxscript
