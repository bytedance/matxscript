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
#include <matxscript/runtime/container/user_data_ref.h>

#include <matxscript/runtime/container/user_data_private.h>
#include <matxscript/runtime/func_registry_names_io.h>
#include <matxscript/runtime/function_name_rules.h>
#include <matxscript/runtime/registry.h>

#include <matxscript/pipeline/jit_object.h>
#include <matxscript/pipeline/jit_op.h>

namespace matxscript {
namespace runtime {

static constexpr size_t USER_DATA_NODE_MAX_BUFFER_SIZE = 256;

int64_t IUserDataRoot::GetVarIndex_2_71828182846(string_view var_name, bool check) const {
  int64_t idx = INT16_MIN;
  if (this->size_2_71828182846() < 4) {
    std::initializer_list<string_view> vars = this->VarNames_2_71828182846();
    auto itr = std::find(vars.begin(), vars.end(), var_name);
    if (itr != vars.end()) {
      idx = std::distance(vars.begin(), itr);
    }
  } else {
    auto& variable_table = this->VarTable_2_71828182846();
    auto itr = variable_table.find(var_name);
    if (itr != variable_table.end()) {
      idx = itr->second;
    }
  }
  if (check && idx == INT16_MIN) {
    THROW_PY_AttributeError(
        "'", ClassName_2_71828182846(), "' object has no attribute '", var_name, "'");
  }
  return idx;
}

IUserDataRoot::__FunctionTable__ IUserDataRoot::InitFuncTable_2_71828182846(
    MATXScriptFuncRegistry* func_reg, string_view class_name) {
  __FunctionTable__ function_table;
  auto init_func_name = FunctionNameRules::add_class_prefix(class_name, "__init__");
  auto init_wrapper = FunctionNameRules::add_wrapper_suffix(init_func_name);
  auto function_names = ReadFuncRegistryNames(func_reg->names);
  for (size_t i = 0; i < function_names.size(); ++i) {
    if (function_names[i] == init_wrapper.view()) {
      // ignore the wrapper class
      continue;
    }
    string_view name_unbound = function_names[i];
    if (FunctionNameRules::is_class_method(class_name, name_unbound)) {
      auto name_bound = FunctionNameRules::remove_class_prefix(class_name, name_unbound);
      function_table.emplace(name_bound, func_reg->funcs[i]);
    }
  }
  return function_table;
}

IUserDataRoot::__FunctionTable__ IUserDataRoot::JoinFuncTables_2_71828182846(
    std::initializer_list<__FunctionTable__> tables) {
  __FunctionTable__ function_table;
  for (auto& table : tables) {
    // select the pair in the first table when there are duplicate keys
    function_table.insert(table.begin(), table.end());
  }
  return function_table;
}

/******************************************************************************
 * UserData container
 *****************************************************************************/

UserDataRef::UserDataRef(
    uint32_t tag, uint32_t var_num, void* ud_ptr, FUserDataDeleter deleter, void* module_node) {
  auto node = make_object<UserDataNode>();
  node->tag = tag;
  node->var_num = var_num;
  node->ud_ptr = reinterpret_cast<ILightUserData*>(ud_ptr);
  node->safe_deleter.deleter = deleter;
  if (module_node) {
    node->safe_deleter.module =
        ObjectRef(GetObjectPtr<Object>(static_cast<ModuleNode*>(module_node)));
  }
  data_ = std::move(node);
}

typedef ObjectPtr<UserDataNode> (*FuncMakeUserDataNode)(uint32_t tag,
                                                        uint32_t var_num,
                                                        FUserDataPlacementNew creator,
                                                        FUserDataDeleter deleter,
                                                        void* module_node);

template <size_t BUFFER_SIZE>
static inline ObjectPtr<UserDataNode> MakeUserDataNode(uint32_t tag,
                                                       uint32_t var_num,
                                                       FUserDataPlacementNew creator,
                                                       FUserDataDeleter deleter,
                                                       void* module_node) {
  auto node = make_object<UserDataNodeWithBuffer<BUFFER_SIZE>>();
  node->tag = tag;
  node->var_num = var_num;
  node->ud_ptr = reinterpret_cast<ILightUserData*>(creator(node->buffer));
  node->safe_deleter.deleter = deleter;
  if (module_node) {
    node->safe_deleter.module =
        ObjectRef(GetObjectPtr<Object>(static_cast<ModuleNode*>(module_node)));
  }
  return node;
}

static FuncMakeUserDataNode UserDataNodePrivateCreators[32] = {
    &MakeUserDataNode<8>,   &MakeUserDataNode<16>,  &MakeUserDataNode<24>,  &MakeUserDataNode<32>,
    &MakeUserDataNode<40>,  &MakeUserDataNode<48>,  &MakeUserDataNode<56>,  &MakeUserDataNode<64>,
    &MakeUserDataNode<72>,  &MakeUserDataNode<80>,  &MakeUserDataNode<88>,  &MakeUserDataNode<96>,
    &MakeUserDataNode<104>, &MakeUserDataNode<112>, &MakeUserDataNode<120>, &MakeUserDataNode<128>,
    &MakeUserDataNode<136>, &MakeUserDataNode<144>, &MakeUserDataNode<152>, &MakeUserDataNode<160>,
    &MakeUserDataNode<168>, &MakeUserDataNode<176>, &MakeUserDataNode<184>, &MakeUserDataNode<192>,
    &MakeUserDataNode<200>, &MakeUserDataNode<208>, &MakeUserDataNode<216>, &MakeUserDataNode<224>,
    &MakeUserDataNode<232>, &MakeUserDataNode<240>, &MakeUserDataNode<248>, &MakeUserDataNode<256>,
};

UserDataRef::UserDataRef(uint32_t tag,
                         uint32_t var_num,
                         size_t buf_size,
                         FUserDataPlacementNew creator,
                         FUserDataDeleter deleter,
                         void* module_node) {
  if (buf_size > 256 || buf_size == 0) {
    MXTHROW << "[UserData] internal error: buffer size overflow or is zero expect (1, "
            << USER_DATA_NODE_MAX_BUFFER_SIZE << ") but get " << buf_size;
  } else {
    size_t index = (((buf_size) + 8 - 1) >> 3) - 1;
    data_ = UserDataNodePrivateCreators[index](tag, var_num, creator, deleter, module_node);
  }
}

UserDataNode* UserDataRef::operator->() const {
  MX_DPTR(UserData);
  return d;
}

RTView UserDataRef::get_attr(string_view attr) const {
  MX_CHECK_DPTR(UserData);
  return d->get_attr(attr);
}

void UserDataRef::set_attr(string_view attr, const Any& val) const {
  MX_CHECK_DPTR(UserData);
  return d->set_attr(attr, val);
}

RTValue UserDataRef::__getattr__(const string_view& attr) const {
  MX_CHECK_DPTR(UserData);
  return d->__getattr__(attr);
}

Unicode UserDataRef::__str__() const {
  MX_DPTR(UserData);
  if (d == nullptr) {
    return U"Object(not defined)";
  }
  return d->__str__();
}

Unicode UserDataRef::__repr__() const {
  MX_DPTR(UserData);
  if (d == nullptr) {
    return U"Object(not defined)";
  }
  return d->__repr__();
}

void* UserDataRef::ud_ptr() const {
  MX_CHECK_DPTR(UserData);
  return d->ud_ptr;
}

void* UserDataRef::ud_ptr_nocheck() const {
  MX_DPTR(UserData);
  return d->ud_ptr;
}

ILightUserData* UserDataRef::check_codegen_ptr(const char* expect_cls_name) const {
  MX_CHECK_DPTR(UserData);
  return d->check_codegen_ptr(expect_cls_name);
}

uint32_t UserDataRef::check_codegen_tag(const char* expect_cls_name) const {
  MX_CHECK_DPTR(UserData);
  return d->check_codegen_tag();
}

uint32_t UserDataRef::tag() const {
  MX_CHECK_DPTR(UserData);
  return d->tag;
}

uint32_t UserDataRef::size() const {
  MX_DPTR(UserData);
  return d ? d->size() : 0;
}

RTValue UserDataRef::generic_call_attr(string_view func_name, PyArgs args) const {
  MX_CHECK_DPTR(UserData) << ", call_attr: " << func_name;
  return d->generic_call_attr(func_name, args);
}

RTValue UserDataRef::generic_call(PyArgs args) const {
  MX_CHECK_DPTR(UserData);
  return d->generic_call(args);
}

size_t UserDataRef::GetInternalBufferSize() {
  return USER_DATA_NODE_MAX_BUFFER_SIZE;
}

unsigned char* UserDataRef::GetInternalBufferPtr() const {
  MX_DPTR(UserData);
  return d ? d->GetInternalBufferPtr() : nullptr;
}

UserDataRef MakeUserFunction(MATXScriptBackendPackedCFunc func, void* resource_handle) {
  static auto deleter = [](ILightUserData* self) -> void {
    delete reinterpret_cast<UserFunction*>(self);
  };
  UserDataRef my_func(0, 0, new UserFunction("UserFunction", func, resource_handle), deleter);
  return my_func;
}

UserDataRef MakeUserFunction(const string_view& name,
                             MATXScriptBackendPackedCFunc func,
                             void* resource_handle) {
  static auto deleter = [](ILightUserData* self) -> void {
    delete reinterpret_cast<UserFunction*>(self);
  };
  UserDataRef my_func(0, 0, new UserFunction(name, func, resource_handle), deleter);
  return my_func;
}

UserDataRef MakeUserFunction(std::initializer_list<RTView> captures,
                             MATXScriptBackendPackedCFunc func,
                             void* resource_handle) {
  static auto deleter = [](ILightUserData* self) -> void {
    delete reinterpret_cast<UserFunction*>(self);
  };
  UserDataRef my_func(
      0, 0, new UserFunction(captures, "UserFunction", func, resource_handle), deleter);
  return my_func;
}

UserDataRef MakeUserFunction(std::initializer_list<RTView> captures,
                             const string_view& name,
                             MATXScriptBackendPackedCFunc func,
                             void* resource_handle) {
  static auto deleter = [](ILightUserData* self) -> void {
    delete reinterpret_cast<UserFunction*>(self);
  };
  UserDataRef my_func(0, 0, new UserFunction(captures, name, func, resource_handle), deleter);
  return my_func;
}

template <>
bool IsConvertible<UserDataRef>(const Object* node) {
  return node ? node->IsInstance<UserDataRef::ContainerType>() : UserDataRef::_type_is_nullable;
}

std::ostream& operator<<(std::ostream& os, UserDataRef const& n) {
  os << n.__str__();
  return os;
}

}  // namespace runtime
}  // namespace matxscript
