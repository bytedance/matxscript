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
#include <gtest/gtest.h>

#include <matxscript/runtime/codegen_all_includes.h>
#include <iostream>

namespace {
using namespace ::matxscript::runtime;

struct UserCustomClass : public IUserDataRoot {
  // flags for convert check
  static uint32_t tag_s_2_71828182846_;
  static uint32_t var_num_s_2_71828182846_;
  static string_view class_name_s_2_71828182846_;
  static IUserDataRoot::__FunctionTable__ function_table_s_2_71828182846_;

  // override meta functions
  const char* ClassName_2_71828182846() const override {
    return "UserCustomClass";
  }
  uint32_t tag_2_71828182846() const override {
    return tag_s_2_71828182846_;
  }
  uint32_t size_2_71828182846() const override {
    return var_num_s_2_71828182846_;
  }

  std::initializer_list<string_view> VarNames_2_71828182846() const override {
    static std::initializer_list<string_view> __var_names_s__ = {
        "name",
        "age",
    };
    return __var_names_s__;
  }

  const ska::flat_hash_map<string_view, int64_t>& VarTable_2_71828182846() const override {
    static ska::flat_hash_map<string_view, int64_t> __var_table_s__ = {
        {"name", 0},
        {"age", 1},
    };
    return __var_table_s__;
  }

  // member vars
  Unicode name;
  int64_t age;

  // override __getitem__ functions
  RTView GetVar_2_71828182846(int64_t idx) const override {
    switch (idx) {
      case 0: {
        return name;
      } break;
      case 1: {
        return age;
      } break;
      default: {
        MXCHECK(false) << "index overflow";
        return nullptr;
      } break;
    }
  }
  // override __setitem__ functions
  void SetVar_2_71828182846(int64_t idx, const Any& val) override {
    switch (idx) {
      case 0: {
        name = val.As<Unicode>();
      } break;
      case 1: {
        age = val.As<int64_t>();
      } break;
      default: {
        MXTHROW << "index overflow";
      } break;
    }
  }
};

// flags for convert check
uint32_t UserCustomClass::tag_s_2_71828182846_ = 0;
uint32_t UserCustomClass::var_num_s_2_71828182846_ = 2;
string_view UserCustomClass::class_name_s_2_71828182846_ = "UserCustomClass";
IUserDataRoot::__FunctionTable__ UserCustomClass::function_table_s_2_71828182846_;

struct UserCustomClassView : public IUserDataSharedViewRoot {
  // member var
  UserCustomClass* ptr;
  // constructor
  UserCustomClassView(UserCustomClass* ptr) : ptr(ptr), IUserDataSharedViewRoot() {
  }
  UserCustomClassView(UserCustomClass* ptr, UserDataRef ref)
      : ptr(ptr), IUserDataSharedViewRoot(std::move(ref)) {
  }
  UserCustomClassView() : ptr(nullptr) {
  }
  UserCustomClassView(const Any& ref) : UserCustomClassView(ref.AsObjectRef<UserDataRef>()) {
  }
  // UserDataRef
  UserCustomClassView(UserDataRef ref) {
    MXCHECK_EQ(UserCustomClass::tag_s_2_71828182846_, ref.tag());
    MXCHECK_EQ(UserCustomClass::var_num_s_2_71828182846_, ref.size());
    ptr = (UserCustomClass*)(ref.check_codegen_ptr());
    ud_ref = std::move(ref);
  }
  const UserCustomClass* operator->() const {
    return ptr;
  }
  UserCustomClass* operator->() {
    return ptr;
  }
  bool operator==(const Any& o) const {
    return ArithOps::eq(RTView(ud_ref), o);
  }
  bool operator!=(const Any& o) const {
    return ArithOps::ne(RTView(ud_ref), o);
  }

  template <typename T,
            typename = typename std::enable_if<std::is_convertible<UserDataRef, T>::value>::type>
  operator T() const {
    return ud_ref;
  }
};

MATX_DLL RTValue UserCustomClass__F___init__(UserCustomClassView self, Unicode name, int64_t age) {
  self->name = name;
  self->age = age;
  return (None);
}

void UserCustomClass_F__deleter__(ILightUserData* ptr) {
  delete static_cast<UserCustomClass*>(ptr);
}
void* UserCustomClass_F__placement_new__(void* buf) {
  return new (buf) UserCustomClass;
}
void UserCustomClass_F__placement_del__(ILightUserData* ptr) {
  static_cast<UserCustomClass*>(ptr)->UserCustomClass::~UserCustomClass();
}

UserDataRef UserCustomClass__F___init___wrapper(Unicode name, int64_t age) {
  static auto buffer_size = UserDataRef::GetInternalBufferSize();
  if (buffer_size < sizeof(UserCustomClass)) {
    auto self = new UserCustomClass;
    UserCustomClass__F___init__(self, name, age);
    self->function_table_2_71828182846_ = &UserCustomClass::function_table_s_2_71828182846_;
    return UserDataRef(UserCustomClass::tag_s_2_71828182846_,
                       UserCustomClass::var_num_s_2_71828182846_,
                       self,
                       UserCustomClass_F__deleter__,
                       nullptr);
  } else {
    UserDataRef self(UserCustomClass::tag_s_2_71828182846_,
                     UserCustomClass::var_num_s_2_71828182846_,
                     sizeof(UserCustomClass),
                     UserCustomClass_F__placement_new__,
                     UserCustomClass_F__placement_del__,
                     nullptr);
    UserCustomClassView self_view(self);
    self_view->function_table_2_71828182846_ = &UserCustomClass::function_table_s_2_71828182846_;
    UserCustomClass__F___init__(self_view, name, age);
    return self;
  }
}

}  // namespace

namespace matxscript {
namespace runtime {

TEST(UserStruct, Converter) {
  UserCustomClassView my_cls = UserCustomClass__F___init___wrapper(U"xx", 16);
  {
    GenericValueConverter<RTValue> Converter;
    RTValue rt_val1 = Converter(my_cls);
    RTValue rt_val2(Converter(my_cls));
  }

  {
    RTView rt_val1 = my_cls;
    // TODO: fixme
    // RTView rt_val2(my_cls);
  }

  {
    GenericValueConverter<RTValue> Converter;
    UserDataRef rt_val1 = Converter(my_cls).As<UserDataRef>();
    UserDataRef rt_val2(Converter(my_cls).As<UserDataRef>());
  }

  {
    GenericValueConverter<RTView> Converter;
    UserDataRef rt_val1 = Converter(my_cls).As<UserDataRef>();
    UserDataRef rt_val2(Converter(my_cls).As<UserDataRef>());
  }

  {
    Dict con;
    con.set_item(Unicode(U"hello"), my_cls);
  }

  {
    List con;
    con.append(my_cls);
  }

  {
    Set con;
    con.add(my_cls);
  }
}

TEST(UserStruct, FTList) {
  FTList<UserCustomClassView> cons;
}

TEST(UserStruct, FTDict) {
  FTDict<int, UserCustomClassView> cons;
}

TEST(UserStruct, FTSet) {
  // not supported
  // FTSet<UserCustomClassView> cons;
  std::cout << "not supported FTSet<UserStruct>" << std::endl;
}

}  // namespace runtime
}  // namespace matxscript
