// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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

/*!
 * Reflection utilities.
 * \file node/reflection.cc
 */
#include <matxscript/ir/_base/reflection.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

// Attr getter.
class AttrGetter : public AttrVisitor {
 public:
  const StringRef& skey;
  RTValue* ret;

  AttrGetter(const StringRef& skey, RTValue* ret) : skey(skey), ret(ret) {
  }

  bool found_ref_object{false};

  void Visit(const char* key, double* value) final {
    if (skey == key)
      *ret = value[0];
  }
  void Visit(const char* key, int64_t* value) final {
    if (skey == key)
      *ret = value[0];
  }
  void Visit(const char* key, uint64_t* value) final {
    MXCHECK_LE(value[0], static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        << "cannot return too big constant";
    if (skey == key)
      *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, int* value) final {
    if (skey == key)
      *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, bool* value) final {
    if (skey == key)
      *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, void** value) final {
    if (skey == key)
      *ret = static_cast<void*>(value[0]);
  }
  void Visit(const char* key, DataType* value) final {
    if (skey == key) {
      *ret = value[0];
    }
  }
  void Visit(const char* key, std::string* value) final {
    if (skey == key)
      *ret = String(value[0].data(), value[0].size());
  }

  void Visit(const char* key, String* value) final {
    if (skey == key)
      *ret = value[0];
  }

  void Visit(const char* key, Unicode* value) final {
    if (skey == key)
      *ret = value[0];
  }

  void Visit(const char* key, NDArray* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
  void Visit(const char* key, ObjectRef* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
};

RTValue ReflectionVTable::GetAttr(Object* self, const StringRef& field_name) const {
  RTValue ret;
  AttrGetter getter(field_name, &ret);

  bool success;
  if (getter.skey == "type_key") {
    ret = self->GetTypeKey();
    success = true;
  } else {
    VisitAttrs(self, &getter);
    success = getter.found_ref_object || ret.type_code() != TypeIndex::kRuntimeNullptr;
  }
  return Tuple::dynamic(success, std::move(ret));
}

// List names;
class AttrDir : public AttrVisitor {
 public:
  std::vector<String>* names;

  void Visit(const char* key, double* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, int64_t* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, uint64_t* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, bool* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, int* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, void** value) final {
    names->push_back(key);
  }
  void Visit(const char* key, DataType* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, std::string* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, String* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, Unicode* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, NDArray* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, ObjectRef* value) final {
    names->push_back(key);
  }
};

std::vector<String> ReflectionVTable::ListAttrNames(Object* self) const {
  std::vector<String> names;
  AttrDir dir;
  dir.names = &names;
  VisitAttrs(self, &dir);
  return names;
}

ReflectionVTable* ReflectionVTable::Global() {
  static ReflectionVTable inst;
  return &inst;
}

ObjectPtr<Object> ReflectionVTable::CreateInitObject(const String& type_key,
                                                     const String& repr_bytes) const {
  uint32_t tindex = Object::TypeKey2Index(type_key);
  if (tindex >= fcreate_.size() || fcreate_[tindex] == nullptr) {
    MXLOG(FATAL) << "TypeError: " << type_key
                 << " is not registered via MATXSCRIPT_REGISTER_NODE_TYPE";
  }
  return fcreate_[tindex](repr_bytes);
}

class NodeAttrSetter : public AttrVisitor {
 public:
  String type_key;
  std::unordered_map<String, RTValue> attrs;

  void Visit(const char* key, double* value) final {
    *value = GetAttr(key).As<double>();
  }
  void Visit(const char* key, int64_t* value) final {
    *value = GetAttr(key).As<int64_t>();
  }
  void Visit(const char* key, uint64_t* value) final {
    *value = GetAttr(key).As<uint64_t>();
  }
  void Visit(const char* key, int* value) final {
    *value = GetAttr(key).As<int>();
  }
  void Visit(const char* key, bool* value) final {
    *value = GetAttr(key).As<bool>();
  }
  void Visit(const char* key, std::string* value) final {
    *value = GetAttr(key).As<String>();
  }
  void Visit(const char* key, String* value) final {
    *value = GetAttr(key).As<String>();
  }
  void Visit(const char* key, Unicode* value) final {
    *value = GetAttr(key).As<Unicode>();
  }
  void Visit(const char* key, void** value) final {
    *value = GetAttr(key).As<void*>();
  }
  void Visit(const char* key, DataType* value) final {
    *value = GetAttr(key).As<DataType>();
  }
  void Visit(const char* key, NDArray* value) final {
    *value = GetAttr(key).AsObjectRef<NDArray>();
  }
  void Visit(const char* key, ObjectRef* value) final {
    *value = GetAttr(key).As<ObjectRef>();
  }

 private:
  RTValue GetAttr(const char* key) {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      MXLOG(FATAL) << type_key << ": require field " << key;
    }
    RTValue v = it->second;
    attrs.erase(it);
    return v;
  }
};

static void InitNodeByPackedArgs(ReflectionVTable* reflection, Object* n, const PyArgs& args) {
  NodeAttrSetter setter;
  setter.type_key = n->GetTypeKey();
  MXCHECK_EQ(args.size() % 2, 0);
  for (int i = 0; i < args.size(); i += 2) {
    setter.attrs.emplace(args[i].As<String>(), args[i + 1].As<RTValue>());
  }
  reflection->VisitAttrs(n, &setter);

  if (setter.attrs.size() != 0) {
    std::ostringstream os;
    os << setter.type_key << " does not contain field ";
    for (const auto& kv : setter.attrs) {
      os << " " << kv.first;
    }
    MXLOG(FATAL) << os.str();
  }
}

ObjectRef ReflectionVTable::CreateObject(const String& type_key, const PyArgs& kwargs) {
  ObjectPtr<Object> n = this->CreateInitObject(type_key);
  InitNodeByPackedArgs(this, n.get(), kwargs);
  return ObjectRef(n);
}

ObjectRef ReflectionVTable::CreateObject(const String& type_key,
                                         const Map<StringRef, ObjectRef>& kwargs) {
  // Redirect to the TArgs version
  // It is not the most efficient way, but CreateObject is not meant to be used
  // in a fast code-path and is mainly reserved as a flexible API for frontends.
  std::vector<RTValue> values;
  values.reserve(kwargs.size() * 2);

  for (const auto& kv : *static_cast<const MapNode*>(kwargs.get())) {
    values.push_back(kv.first);
    values.push_back(kv.second);
  }

  return CreateObject(type_key, PyArgs(values.data(), values.size()));
}

// Expose to FFI APIs.
static RTValue NodeGetAttr(PyArgs args) {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* self = static_cast<Object*>(args[0].value().data.v_handle);
  return ReflectionVTable::Global()->GetAttr(self, args[1].As<StringRef>());
}

static RTValue NodeListAttrNames(PyArgs args) {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* self = static_cast<Object*>(args[0].value().data.v_handle);
  auto attr_names = ReflectionVTable::Global()->ListAttrNames(self);
  return Tuple(std::make_move_iterator(attr_names.begin()),
               std::make_move_iterator(attr_names.end()));
}

// API function to make node.
// args format:
//   key1, value1, ..., key_n, value_n
RTValue MakeNode(PyArgs args) {
  String type_key = args[0].As<String>();
  String empty_str;
  PyArgs kwargs(args.begin() + 1, args.size() - 1);
  return ReflectionVTable::Global()->CreateObject(type_key, kwargs);
}

MATXSCRIPT_REGISTER_GLOBAL("runtime.NodeGetAttr").set_body(NodeGetAttr);

MATXSCRIPT_REGISTER_GLOBAL("runtime.NodeListAttrNames").set_body(NodeListAttrNames);

MATXSCRIPT_REGISTER_GLOBAL("runtime.MakeNode").set_body(MakeNode);
}  // namespace runtime
}  // namespace matxscript
