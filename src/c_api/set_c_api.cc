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

MATXSCRIPT_REGISTER_GLOBAL("runtime.Set").set_body([](PyArgs args) -> RTValue {
  Set data;
  for (int i = 0; i < args.size(); ++i) {
    data.emplace(args[i].As<RTValue>());
  }
  return data;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.FTSet").set_body([](PyArgs args) -> RTValue {
  FTSet<RTValue> data;
  for (int i = 0; i < args.size(); ++i) {
    data.emplace(args[i].As<RTValue>());
  }
  return data;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetEqual").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "set.__eq__ expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  const auto& other = args[1];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectRefNoCheck<Set>() == other.AsObjectRefNoCheck<Set>();
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__eq__",
                                                                       {other.As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Set_Iter").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "set.__iter__ expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectRefNoCheck<Set>().iter();
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__iter__", {});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetSize").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "set.__len__ expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectRefNoCheck<Set>().size();
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__len__", {});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetContains").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "set.__contains__ expect " << 2 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  const auto& key = args[1];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectRefNoCheck<Set>().contains(key.As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__contains__",
                                                                       {key.As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetAddItem").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "set.add expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  const auto& key = args[1];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      self.AsObjectRefNoCheck<Set>().add(key.As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTSet: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("add", {key.As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetClear").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "set.clear expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      self.AsObjectRefNoCheck<Set>().clear();
    } break;
    case TypeIndex::kRuntimeFTSet: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("clear", {});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetReserve").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "set.reserve expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      self.AsObjectRefNoCheck<Set>().reserve(args[1].As<int64_t>());
    } break;
    case TypeIndex::kRuntimeFTSet: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("reserve", {args[1].As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetBucketCount").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 == args.size()) << "set.bucket_count expect " << 1 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectRefNoCheck<Set>().bucket_count();
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("bucket_count", {});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetDifference").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 <= args.size()) << "set.difference expect no less than " << 1 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  PyArgs params = PyArgs(args.begin() + 1, args.size() - 1);

  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectRefNoCheck<Set>().difference(params);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("difference", params);
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetDifferenceUpdate").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 <= args.size()) << "set.difference_update expect no less than " << 1
                            << " arguments but get " << args.size();
  const auto& self = args[0];
  PyArgs params = PyArgs(args.begin() + 1, args.size() - 1);

  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      self.AsObjectRefNoCheck<Set>().difference_update(params);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("difference_update", params);
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetDiscard").set_body([](PyArgs args) -> RTValue {
  MXCHECK(2 == args.size()) << "set.discard expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      self.AsObjectRefNoCheck<Set>().discard(args[1].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTSet: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("discard", {args[1].As<RTView>()});
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetUpdate").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 <= args.size()) << "set.update expect no less than " << 1 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  PyArgs params = PyArgs(args.begin() + 1, args.size() - 1);

  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      self.AsObjectRefNoCheck<Set>().update(params);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("update", params);
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.SetUnion").set_body([](PyArgs args) -> RTValue {
  MXCHECK(1 <= args.size()) << "set.union expect no less than " << 1 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  PyArgs params = PyArgs(args.begin() + 1, args.size() - 1);

  switch (self.type_code()) {
    case TypeIndex::kRuntimeSet: {
      return self.AsObjectRefNoCheck<Set>().set_union(params);
    } break;
    case TypeIndex::kRuntimeFTSet: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("union", params);
    } break;
    default: {
      MXTHROW << "expect 'set' but get '" << self.type_name();
    } break;
  }
  return None;
});

}  // namespace runtime
}  // namespace matxscript
