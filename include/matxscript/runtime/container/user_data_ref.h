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
#pragma once

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <matxscript/runtime/c_backend_api.h>
#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/user_data_interface.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

struct IUserDataRoot : public ILightUserData {
 public:
  using __FunctionTable__ = ska::flat_hash_map<string_view, MATXScriptBackendPackedCFunc>;
  ~IUserDataRoot() override = default;
  IUserDataRoot() = default;

  int32_t type_2_71828182846() const override {
    return UserDataStructType::kUserData;
  }

  // var names
  virtual std::initializer_list<string_view> VarNames_2_71828182846() const {
    return std::initializer_list<string_view>{};
  }
  // var table
  virtual const ska::flat_hash_map<string_view, int64_t>& VarTable_2_71828182846() const {
    static ska::flat_hash_map<string_view, int64_t> empty_vars{};
    return empty_vars;
  }
  // instance check
  virtual bool isinstance_2_71828182846(uint64_t tag) {
    return true;
  }

  // visit member var
  virtual RTView GetVar_2_71828182846(int64_t idx) const = 0;
  virtual void SetVar_2_71828182846(int64_t idx, const Any& val) = 0;

  int64_t GetVarIndex_2_71828182846(string_view var_name, bool check = true) const;

  inline RTView __getattr__(string_view var_name) const override {
    return GetVar_2_71828182846(GetVarIndex_2_71828182846(var_name, true));
  }

  inline void __setattr__(string_view var_name, const Any& val) override {
    return SetVar_2_71828182846(GetVarIndex_2_71828182846(var_name, true), val);
  }

  // function table
  static __FunctionTable__ InitFuncTable_2_71828182846(MATXScriptFuncRegistry* func_reg,
                                                       string_view class_name);
  // join multi function table
  // select the pair in the first table when there are duplicate keys
  static __FunctionTable__ JoinFuncTables_2_71828182846(
      std::initializer_list<__FunctionTable__> tables);

  // function table is unbound
  __FunctionTable__* function_table_2_71828182846_ = nullptr;
};

template <typename FROM_TYPE, typename TO_TYPE>
MATXSCRIPT_ALWAYS_INLINE TO_TYPE CAST_TO_CLASS_VIEW_NOCHECK(const FROM_TYPE& o) {
  return TO_TYPE(o.ptr, o.ud_ref);
}
template <typename FROM_TYPE, typename TO_TYPE>
MATXSCRIPT_ALWAYS_INLINE TO_TYPE CAST_TO_CLASS_VIEW(const FROM_TYPE& o) {
  return TO_TYPE(o.ud_ref);
}

/******************************************************************************
 * User Class Example:
 *
 * namespace {
 * struct MyUserStruct : public IUserDataRoot {
 *   // flags for convert check
 *   static uint32_t tag = Hash(ClassName + VarTypeName + VarName);
 *   static uint32_t var_num = 2;
 *   static string_view class_name = "MyUserStruct";
 *
 *   // member vars
 *   Type0 var0;
 *   Type1 var1;
 *   ...
 *
 *   // override virtual functions
 *   const char* ClassName() const override { return "MyUserStruct"; }
 *   ...
 * };
 *
 * struct MyUserStructView {
 *   // member var
 *   MyUserStruct *ptr;
 *   // constructor
 *   MyUserStructView(MyUserStruct* ptr) : ptr(ptr) {
 *   }
 *   // UserDataRef
 *   MyUserStructView(const UserDataRef& ref) {
 *     CHECK_EQ(MyUserStruct::tag, ref->tag);
 *     CHECK_EQ(MyUserStruct::var_num, ref->var_num);
 *     ptr = (MyUserStruct*)(ref->ud_ptr);
 *   }
 *   const MyUserStruct* operator->() const {
 *     return ptr;
 *   }
 *   MyUserStruct* operator->() {
 *     return ptr;
 *   }
 * }
 *
 * // __del__
 * void MyUserStruct_F__deleter__(void* ptr) {
 *   delete static_cast<MyUserStruct*>(ptr);
 * }
 *
 * // MyUserStruct_F__init__(MyUserStructView self, ...) {
 *   ...
 * }
 *
 * // __init__
 * UserDataRef MyUserStruct_F__init__wrapper(...) {
 *   auto self = new MyUserStruct();
 *   MyUserStruct_F__init__(self, ...);
 *   return UserDataRef(MyUserStruct::tag,
 *                      MyUserStruct::var_num,
 *                      self,
 *                      MyUserStruct_F__deleter__);
 * }
 *
 * // member function
 * ReturnType MyUserStruct_F_member_func(MyUserStructView self, ...) {
 *
 * }
 *
 * } // namespace
 *
 * // entry info
 * extern "C" {
 *   int MyUserStruct_F_member_func_c_api(MATXValue* args,
 *                                        int* type_codes,
 *                                        int num_args,
 *                                        MATXValue* out_ret_value,
                                          int* out_ret_tcode,
 *                                        void* resource_handle) {
 *    TArgs args_t(args, type_codes, num_args);
 *    MyUserStruct_F_member_func(UserDataRef(args_t[0]), args_t[1],...);
 *    ...
 *  }
 *
 * } // extern
 *****************************************************************************/

typedef void (*FUserDataDeleter)(ILightUserData* self);
typedef void* (*FUserDataPlacementNew)(void* buf);
class UserDataNode;

/*! \brief reference to user data objects. */
class UserDataRef : public ObjectRef {
 public:
  using ContainerType = UserDataNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

  UserDataNode* operator->() const;

 public:
  UserDataRef() = default;
  UserDataRef(const UserDataRef& other) noexcept = default;
  UserDataRef(UserDataRef&& other) noexcept = default;
  UserDataRef& operator=(const UserDataRef& other) noexcept = default;
  UserDataRef& operator=(UserDataRef&& other) noexcept = default;
  explicit UserDataRef(::matxscript::runtime::ObjectPtr<::matxscript::runtime::Object> n) noexcept
      : ObjectRef(std::move(n)) {
  }
  UserDataRef(uint32_t tag,
              uint32_t var_num,
              void* ud_ptr,
              FUserDataDeleter deleter,
              void* module_node = nullptr);

  UserDataRef(uint32_t tag,
              uint32_t var_num,
              size_t buf_size,
              FUserDataPlacementNew creator,
              FUserDataDeleter deleter,
              void* module_node);

 public:
  // get/set var only
  RTView get_attr(string_view attr) const;
  void set_attr(string_view attr, const Any& val) const;

  // get var or method
  RTValue __getattr__(const string_view& attr) const;

  Unicode __str__() const;

  Unicode __repr__() const;

  // member var num
  uint32_t tag() const;
  uint32_t size() const;
  void* ud_ptr() const;
  void* ud_ptr_nocheck() const;
  ILightUserData* check_codegen_ptr(const char* expect_cls_name = "") const;
  uint32_t check_codegen_tag(const char* expect_cls_name = "") const;

  RTValue generic_call(PyArgs args) const;
  RTValue generic_call_attr(string_view func_name, PyArgs args) const;

  template <typename... ARGS>
  inline RTValue call(ARGS&&... args) const {
    // TODO: check convertible
    GenericValueConverter<RTView> Converter;
    std::initializer_list<RTView> packed_args{Converter(std::forward<ARGS>(args))...};
    return generic_call(packed_args);
  }

  template <typename... ARGS>
  inline RTValue call_attr(string_view func_name, ARGS&&... args) const {
    GenericValueConverter<RTView> Converter;
    std::initializer_list<RTView> packed_args{Converter(std::forward<ARGS>(args))...};
    return generic_call_attr(func_name, packed_args);
  }

  unsigned char* GetInternalBufferPtr() const;
  static size_t GetInternalBufferSize();

 private:
  friend class JitObject;
};

UserDataRef MakeUserFunction(MATXScriptBackendPackedCFunc func, void* resource_handle);
UserDataRef MakeUserFunction(const string_view& name,
                             MATXScriptBackendPackedCFunc func,
                             void* resource_handle);
UserDataRef MakeUserFunction(std::initializer_list<RTView> captures,
                             MATXScriptBackendPackedCFunc func,
                             void* resource_handle);
UserDataRef MakeUserFunction(std::initializer_list<RTView> captures,
                             const string_view& name,
                             MATXScriptBackendPackedCFunc func,
                             void* resource_handle);

template <
    typename UserType,
    typename = typename std::enable_if<std::is_base_of<ILightUserData, UserType>::value>::type>
MATXSCRIPT_ALWAYS_INLINE UserDataRef CAST_TO_USER_DATA_REF(const UserType* o) {
  return UserDataRef(GetObjectPtr<Object>(o->self_node_ptr_2_71828182846));
}

// only for codegen class
struct IUserDataSharedViewRoot {
  UserDataRef ud_ref{ObjectPtr<Object>(nullptr)};
  // constructor
  IUserDataSharedViewRoot(UserDataRef ref) : ud_ref(std::move(ref)) {
  }
  IUserDataSharedViewRoot() : ud_ref(ObjectPtr<Object>(nullptr)) {
  }
};

namespace TypeIndex {
template <>
struct type_index_traits<UserDataRef> {
  static constexpr int32_t value = kRuntimeUserData;
};
}  // namespace TypeIndex

template <>
bool IsConvertible<UserDataRef>(const Object* node);

template <>
MATXSCRIPT_ALWAYS_INLINE UserDataRef Any::As<UserDataRef>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeUserData);
  return UserDataRef(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE UserDataRef Any::AsNoCheck<UserDataRef>() const {
  return UserDataRef(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

std::ostream& operator<<(std::ostream& os, UserDataRef const& n);

}  // namespace runtime
}  // namespace matxscript
