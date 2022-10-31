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
#include <matxscript/runtime/container/user_data_private.h>

#include <matxscript/runtime/container/native_object_private.h>

#include <matxscript/pipeline/jit_object.h>
#include <matxscript/pipeline/jit_op.h>

namespace matxscript {
namespace runtime {

const UserDataNode* UserDataNode::StripJitWrapper(const UserDataNode* node) {
  auto ud_ptr = node->ud_ptr;
  if (MATXSCRIPT_UNLIKELY(ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData)) {
    auto nat_obj_ptr = static_cast<NativeObject*>(ud_ptr);
    if (nat_obj_ptr->is_jit_object_) {
      auto* jit_ptr =
          static_cast<JitObject*>(static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get()));
      return StripJitWrapper(jit_ptr->self().operator->());
    } else if (nat_obj_ptr->is_native_op_) {
      auto* op_ptr = static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get());
      if (op_ptr->ClassName() == "JitOp") {
        auto* jit_op_ptr = static_cast<JitOp*>(op_ptr);
        auto* jit_ptr = static_cast<JitObject*>(jit_op_ptr->jit_object_.get());
        return StripJitWrapper(jit_ptr->self().operator->());
      }
    }
  }
  return node;
}

/******************************************************************************
 * UserDataNode
 *****************************************************************************/
ILightUserData* UserDataNode::check_codegen_ptr(const char* expect_cls_name) const {
  if (MATXSCRIPT_UNLIKELY(ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData)) {
    auto nat_obj_ptr = static_cast<NativeObject*>(ud_ptr);
    if (nat_obj_ptr->is_jit_object_) {
      auto* jit_ptr =
          static_cast<JitObject*>(static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get()));
      return jit_ptr->self().check_codegen_ptr();
    } else if (nat_obj_ptr->is_native_op_) {
      auto* op_ptr = static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get());
      if (op_ptr->ClassName() == "JitOp") {
        auto* jit_op_ptr = static_cast<JitOp*>(op_ptr);
        auto* jit_ptr = static_cast<JitObject*>(jit_op_ptr->jit_object_.get());
        return jit_ptr->self().check_codegen_ptr();
      }
    }
    MXTHROW << "Expect a codegen object '" << expect_cls_name << "', but get '"
            << nat_obj_ptr->native_class_name_ << "'";
    return nullptr;
  } else {
    return ud_ptr;
  }
}

uint32_t UserDataNode::check_codegen_tag(const char* expect_cls_name) const {
  if (MATXSCRIPT_UNLIKELY(ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData)) {
    auto nat_obj_ptr = static_cast<NativeObject*>(ud_ptr);
    if (nat_obj_ptr->is_jit_object_) {
      auto* jit_ptr =
          static_cast<JitObject*>(static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get()));
      return jit_ptr->self().check_codegen_tag();
    } else if (nat_obj_ptr->is_native_op_) {
      auto* op_ptr = static_cast<OpKernel*>(nat_obj_ptr->opaque_ptr_.get());
      if (op_ptr->ClassName() == "JitOp") {
        auto* jit_op_ptr = static_cast<JitOp*>(op_ptr);
        auto* jit_ptr = static_cast<JitObject*>(jit_op_ptr->jit_object_.get());
        return jit_ptr->self().check_codegen_tag();
      }
    }
    MXTHROW << "Expect a codegen object '" << expect_cls_name << "', but get '"
            << nat_obj_ptr->native_class_name_ << "'";
    return tag;
  } else {
    return ud_ptr->tag_2_71828182846();
  }
}

namespace {

inline RTValue call_native(UserDataNode* self, string_view func_name, PyArgs args) {
  auto ud_ptr = (NativeObject*)(self->ud_ptr);
  auto table = ud_ptr->function_table_;
  auto f_table_itr = table->find(func_name);
  if (MATXSCRIPT_UNLIKELY(f_table_itr == table->end())) {
    if (ud_ptr->is_jit_object_) {
      auto* jit_ptr = static_cast<JitObject*>(static_cast<OpKernel*>(ud_ptr->opaque_ptr_.get()));
      return jit_ptr->generic_call_attr(func_name, args);
    } else if (ud_ptr->is_native_op_) {
      auto* op_ptr = static_cast<OpKernel*>(ud_ptr->opaque_ptr_.get());
      if (op_ptr->ClassName() == "JitOp") {
        auto* jit_op_ptr = static_cast<JitOp*>(op_ptr);
        return jit_op_ptr->generic_call_attr(func_name, args);
      }
    }
    MXTHROW << "AttributeError: '" << ud_ptr->native_class_name_ << "' object has no attribute '"
            << func_name << "'";
  }
  return f_table_itr->second(ud_ptr->opaque_ptr_.get(), args);
}

inline RTValue call_native_function(UserDataNode* self, PyArgs args) {
  return (*(((NativeFuncUserData*)(self->ud_ptr))->__call__))(args);
}

inline RTValue call_function(UserDataNode* self, PyArgs args) {
  return ((UserFunction*)(self->ud_ptr))->generic_call(args);
}

inline RTValue call_class_method(UserDataNode* self, string_view func_name, PyArgs args) {
  const int kNumArgs = args.size() + 1;
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  MATXScriptAny values[kArraySize];

  // find method
  auto ud_ptr = ((IUserDataRoot*)(self->ud_ptr));
  auto f_table_itr = ud_ptr->function_table_2_71828182846_->find(func_name);
  if (MATXSCRIPT_UNLIKELY(f_table_itr == ud_ptr->function_table_2_71828182846_->end())) {
    MXTHROW << "AttributeError: '" << ud_ptr->ClassName_2_71828182846()
            << "' object has no attribute '" << func_name << "'";
  }
  MATXScriptBackendPackedCFunc c_packed_func = f_table_itr->second;
  // always bound self
  values[0].data.v_handle = static_cast<Object*>(self);
  values[0].code = TypeIndex::kRuntimeUserData;
  for (size_t i = 0; i < args.size(); ++i) {
    values[i + 1] = args[i].value();
  }
  MATXScriptAny out_ret_value;
  c_packed_func(values, kNumArgs, &out_ret_value, self->ud_ptr);
  return RTValue::MoveFromCHost(&out_ret_value);
}
}  // namespace

RTValue UserDataNode::generic_call_attr(string_view func_name, PyArgs args) {
  switch (ud_ptr->type_2_71828182846()) {
    case UserDataStructType::kFunction: {
      return call_function(this, args);
    } break;
    case UserDataStructType::kNativeData: {
      return call_native(this, func_name, args);
    } break;
    case UserDataStructType::kNativeFunc: {
      return call_native_function(this, args);
    } break;
    case UserDataStructType::kUserData: {
      return call_class_method(this, func_name, args);
    } break;
    default: {
      MXTHROW << "AttributeError: '" << ud_ptr->ClassName_2_71828182846()
              << "' object has no attribute '" << func_name << "'";
    } break;
  }
  return None;
}

RTValue UserDataNode::generic_call(PyArgs args) {
  auto ud_type = ud_ptr->type_2_71828182846();
  if (ud_type == UserDataStructType::kFunction) {
    return call_function(this, args);
  }
  if (ud_type == UserDataStructType::kNativeFunc) {
    return call_native_function(this, args);
  }
  return generic_call_attr("__call__", args);
}

namespace {
struct NativeMethodClosure : NativeFuncUserData {
  NativeMethodClosure() = default;
  ~NativeMethodClosure() override = default;
  std::function<RTValue(PyArgs)> method;
};

void* NativeMethodClosure_Creator(void* data) {
  auto d = new (data) NativeMethodClosure;
  return d;
}

void NativeMethodClosure_Deleter(ILightUserData* data) {
  ((NativeMethodClosure*)(data))->~NativeMethodClosure();
}

UserDataRef get_native_object_method(const UserDataNode* ud, const string_view& name) {
  auto native_ptr = (NativeObject*)(ud->ud_ptr);
  auto fit = native_ptr->function_table_->find(name);
  if (fit == native_ptr->function_table_->end()) {
    THROW_PY_AttributeError(native_ptr->native_class_name_, " has no attribute ", name);
  }
  auto ret = UserDataRef(0,
                         0,
                         sizeof(NativeMethodClosure),
                         NativeMethodClosure_Creator,
                         NativeMethodClosure_Deleter,
                         nullptr);
  ((NativeMethodClosure*)(ret->ud_ptr))->method = [self = native_ptr->opaque_ptr_,
                                                   func = fit->second](PyArgs args) -> RTValue {
    return func(self.get(), args);
  };
  ((NativeMethodClosure*)(ret->ud_ptr))->__call__ = &((NativeMethodClosure*)(ret->ud_ptr))->method;
  return ret;
}

RTValue get_user_class_attr_or_method(const UserDataNode* ud, const string_view& name) {
  auto cls_base_ptr = (IUserDataRoot*)(ud->ud_ptr);
  auto var_idx = cls_base_ptr->GetVarIndex_2_71828182846(name, false);
  if (var_idx >= 0) {
    // visit var
    return cls_base_ptr->GetVar_2_71828182846(var_idx);
  } else {
    // visit method
    auto fit = cls_base_ptr->function_table_2_71828182846_->find(name);
    if (fit == cls_base_ptr->function_table_2_71828182846_->end()) {
      THROW_PY_AttributeError(cls_base_ptr->ClassName_2_71828182846(), " has no attribute ", name);
    }
    auto ret = UserDataRef(0,
                           0,
                           sizeof(NativeMethodClosure),
                           NativeMethodClosure_Creator,
                           NativeMethodClosure_Deleter,
                           nullptr);
    ((NativeMethodClosure*)(ret->ud_ptr))->method =
        [self = UserDataRef(GetObjectPtr<UserDataNode>(const_cast<UserDataNode*>(ud))),
         attr = String(name)](PyArgs args) -> RTValue {
      return self.generic_call_attr(attr, args);
    };
    ((NativeMethodClosure*)(ret->ud_ptr))->__call__ =
        &((NativeMethodClosure*)(ret->ud_ptr))->method;
    return ret;
  }
}
}  // namespace

RTValue UserDataNode::__getattr__(const string_view& attr) const {
  auto deep_node_ptr = StripJitWrapper(this);
  switch (deep_node_ptr->ud_ptr->type_2_71828182846()) {
    case UserDataStructType::kNativeData: {
      return get_native_object_method(deep_node_ptr, attr);
    } break;
    case UserDataStructType::kUserData: {
      return get_user_class_attr_or_method(deep_node_ptr, attr);
    } break;
    default: {
      THROW_PY_AttributeError(ud_ptr->ClassName_2_71828182846(), " has no attribute ", attr);
    } break;
  }
  return None;
}

Unicode UserDataNode::__str__() const {
  std::stringstream os;
  auto deep_node_ptr = StripJitWrapper(this);
  auto mutable_node = const_cast<UserDataNode*>(deep_node_ptr);
  switch (mutable_node->ud_ptr->type_2_71828182846()) {
    case UserDataStructType::kFunction:
    case UserDataStructType::kNativeFunc: {
      os << "<function " << mutable_node->ud_ptr->ClassName_2_71828182846() << " at " << std::hex
         << mutable_node->ud_ptr << ">";
    } break;
    case UserDataStructType::kNativeData: {
      auto nat_obj_ptr = static_cast<NativeObject*>(mutable_node->ud_ptr);
      if (nat_obj_ptr->function_table_->count("__str__")) {
        auto s = mutable_node->generic_call_attr("__str__", {});
        if (!s.Is<Unicode>()) {
          THROW_PY_TypeError("__str__ returned non-string (type bytes)");
        }
        os << s.AsNoCheck<unicode_view>();
      } else if (nat_obj_ptr->function_table_->count("__repr__")) {
        auto s = mutable_node->generic_call_attr("__repr__", {});
        if (!s.Is<Unicode>()) {
          THROW_PY_TypeError("__repr__ returned non-string (type bytes)");
        }
        os << s.AsNoCheck<unicode_view>();
      } else {
        os << "<" << nat_obj_ptr->native_class_name_ << " object at " << std::hex
           << nat_obj_ptr->opaque_ptr_.get() << ">";
      }
    } break;
    case UserDataStructType::kUserData: {
      auto ud_ptr = ((IUserDataRoot*)(mutable_node->ud_ptr));
      if (ud_ptr->function_table_2_71828182846_->count("__str__")) {
        auto s = mutable_node->generic_call_attr("__str__", {});
        if (!s.Is<Unicode>()) {
          THROW_PY_TypeError("__str__ returned non-string (type bytes)");
        }
        os << s.AsNoCheck<unicode_view>();
      } else if (ud_ptr->function_table_2_71828182846_->count("__repr__")) {
        auto s = mutable_node->generic_call_attr("__repr__", {});
        if (!s.Is<Unicode>()) {
          THROW_PY_TypeError("__repr__ returned non-string (type bytes)");
        }
        os << s.AsNoCheck<unicode_view>();
      } else {
        os << "<" << mutable_node->ud_ptr->ClassName_2_71828182846() << " object at " << std::hex
           << mutable_node->ud_ptr << ">";
      }
    } break;
    default: {
      os << mutable_node->ud_ptr->ClassName_2_71828182846() << "(addr:" << mutable_node->ud_ptr
         << ")";
    } break;
  }
  auto object_str = os.str();
  return UTF8Decode(object_str.data(), object_str.size());
}

Unicode UserDataNode::__repr__() const {
  std::stringstream os;
  auto deep_node_ptr = StripJitWrapper(this);
  auto mutable_node = const_cast<UserDataNode*>(deep_node_ptr);
  switch (mutable_node->ud_ptr->type_2_71828182846()) {
    case UserDataStructType::kFunction:
    case UserDataStructType::kNativeFunc: {
      os << "<function " << mutable_node->ud_ptr->ClassName_2_71828182846() << " at " << std::hex
         << mutable_node->ud_ptr << ">";
    } break;
    case UserDataStructType::kNativeData: {
      auto nat_obj_ptr = static_cast<NativeObject*>(mutable_node->ud_ptr);
      if (nat_obj_ptr->function_table_->count("__repr__")) {
        auto s = mutable_node->generic_call_attr("__repr__", {});
        if (!s.Is<Unicode>()) {
          THROW_PY_TypeError("__repr__ returned non-string (type bytes)");
        }
        os << s.AsNoCheck<unicode_view>();
      } else {
        os << "<" << nat_obj_ptr->native_class_name_ << " object at " << std::hex
           << nat_obj_ptr->opaque_ptr_.get() << ">";
      }
    } break;
    case UserDataStructType::kUserData: {
      auto ud_ptr = ((IUserDataRoot*)(mutable_node->ud_ptr));
      if (ud_ptr->function_table_2_71828182846_->count("__repr__")) {
        auto s = mutable_node->generic_call_attr("__repr__", {});
        if (!s.Is<Unicode>()) {
          THROW_PY_TypeError("__repr__ returned non-string (type bytes)");
        }
        os << s.AsNoCheck<unicode_view>();
      } else {
        os << "<" << mutable_node->ud_ptr->ClassName_2_71828182846() << " object at " << std::hex
           << mutable_node->ud_ptr << ">";
      }
    } break;
    default: {
      os << mutable_node->ud_ptr->ClassName_2_71828182846() << "(addr:" << mutable_node->ud_ptr
         << ")";
    } break;
  }
  auto object_str = os.str();
  return UTF8Decode(object_str.data(), object_str.size());
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(UserDataNode);
}  // namespace runtime
}  // namespace matxscript
