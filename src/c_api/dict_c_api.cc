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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Dict container
 *****************************************************************************/

MATXSCRIPT_REGISTER_GLOBAL("runtime.Dict").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size() % 2, 0);
  Dict data;
  for (int i = 0; i < args.size(); i += 2) {
    data.emplace(args[i].As<RTValue>(), args[i + 1].As<RTValue>());
  }
  return data;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.FTDict").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size() % 2, 0);
  FTDict<RTValue, RTValue> data;
  for (int i = 0; i < args.size(); i += 2) {
    data.emplace(args[i].As<RTValue>(), args[i + 1].As<RTValue>());
  }
  return data;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictEqual").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "dict.__eq__ expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  const auto& other = args[1];

  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      switch (other.type_code()) {
        case TypeIndex::kRuntimeDict: {
          return self.AsObjectRefNoCheck<Dict>() == other.AsObjectRefNoCheck<Dict>();
        } break;
        case TypeIndex::kRuntimeFTDict: {
          return other.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__eq__",
                                                                            {self.As<RTView>()});
        } break;
      }
    } break;
    case TypeIndex::kRuntimeFTDict: {
      switch (other.type_code()) {
        case TypeIndex::kRuntimeDict:
        case TypeIndex::kRuntimeFTDict: {
          return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__eq__",
                                                                           {other.As<RTView>()});
        } break;
      }
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return false;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Dict_Iter").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "dict.__iter__ expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectRefNoCheck<Dict>().key_iter();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__iter__", {});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Dict_KeyIter").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "dict.keys expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectRefNoCheck<Dict>().key_iter();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("keys", {});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Dict_ValueIter").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "dict.values expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectRefNoCheck<Dict>().value_iter();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("values", {});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Dict_ItemIter").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "dict.items expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectRefNoCheck<Dict>().item_iter();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("items", {});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictSize").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "dict.__len__ expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectRefNoCheck<Dict>().size();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__len__", {});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictContains").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "dict.__contains__ expect " << 2 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectRefNoCheck<Dict>().contains(args[1].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__contains__",
                                                                       {args[1].As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return false;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictGetItem").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "dict.__getitem__ expect " << 2 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  const auto& key = args[1];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      auto obj = self.AsObjectRefNoCheck<Dict>();
      MXCHECK(obj.contains(key.As<RTValue>())) << "cannot find the corresponding key in the Dict";
      return obj[key.As<RTValue>()];
    } break;
    case TypeIndex::kRuntimeFTDict: {
      auto obj = self.AsObjectRefNoCheck<FTObjectBase>();
      MXCHECK(obj.generic_call_attr("__contains__", {key.As<RTView>()}).As<bool>())
          << "cannot find the corresponding key in the Dict";
      return obj.generic_call_attr("__getitem__", {key.As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictClear").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "dict.clear expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      self.AsObjectRefNoCheck<Dict>().clear();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("clear", {});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictSetItem").set_body([](PyArgs args) -> RTValue {
  const auto& key = args[1];
  const auto& val = args[2];

  MXCHECK(3 == args.size()) << "dict.__contains__ expect " << 3 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      self.AsObjectRefNoCheck<Dict>()[key.As<RTValue>()] = val.As<RTValue>();
    } break;
    case TypeIndex::kRuntimeFTDict: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr(
          "__setitem__", {key.As<RTView>(), val.As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'dict' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictReserve").set_body([](PyArgs args) -> RTValue {
  Dict obj = args[0].As<Dict>();
  obj.reserve(args[1].As<int64_t>());
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictBucketCount").set_body([](PyArgs args) -> RTValue {
  Dict obj = args[0].As<Dict>();
  return static_cast<int64_t>(obj.bucket_count());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictGetDefault").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size() || 3 == args.size())
      << "dict.get expect " << 2 << " or " << 3 << " arguments but get " << args.size();
  const auto& self = args[0];
  const auto& key = args[1];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      if (args.size() == 2) {
        return self.AsObjectRefNoCheck<Dict>().get_default(key.As<RTValue>(), None);
      } else {
        return self.AsObjectRefNoCheck<Dict>().get_default(key.As<RTValue>(),
                                                           args[2].As<RTValue>());
      }
    } break;
    case TypeIndex::kRuntimeFTDict: {
      if (args.size() == 2) {
        return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("get", {key.As<RTView>()});
      } else {
        return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr(
            "get", {key.As<RTView>(), args[2].As<RTView>()});
      }
    } break;
    default: {
      MXTHROW << "expect 'dict' bug get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DictPop").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size() || 3 == args.size())
      << "dict.pop expect " << 2 << " or " << 3 << " arguments but get " << args.size();

  const auto& self = args[0];
  PyArgs params = PyArgs(args.begin() + 1, args.size() - 1);

  switch (self.type_code()) {
    case TypeIndex::kRuntimeDict: {
      return self.AsObjectRefNoCheck<Dict>().pop(params);
    } break;
    case TypeIndex::kRuntimeFTDict: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("pop", params);
    } break;
    default: {
      MXTHROW << "expect 'dict' bug get '" << self.type_name();
    } break;
  }
  return None;
});

}  // namespace runtime
}  // namespace matxscript
