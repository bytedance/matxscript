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
#include "userdata_mutator.h"

#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/container_private.h>

namespace matxscript {
namespace runtime {

void UserDataMutator::Mutate(RTValue* val, const std::function<void(RTValue*)>& func) {
  // only convert List/Dict/ADT
  switch (val->type_code()) {
    case TypeIndex::kRuntimeList: {
      for (auto& item : val->AsObjectRef<List>()) {
        Mutate(&item, func);
      }
    } break;
    case TypeIndex::kRuntimeDict: {
      auto d = val->AsObjectRef<Dict>();
      for (auto itr = d.item_begin(); itr != d.item_end(); ++itr) {
        Mutate(&itr->second, func);
      }
    } break;
    case TypeIndex::kRuntimeTuple: {
      auto adt = val->AsObjectRef<Tuple>();
      for (auto i = 0; i < adt.size(); ++i) {
        Mutate(&adt[i], func);
      }
    } break;
    default: {
      return func(val);
    } break;
  }
}

void UserDataMutator::Mutate(RTValue* val, OpKernel* op_ptr) {
  auto Converter = [op_ptr](RTValue* val) -> void {
    if ((!val->is_nullptr()) && val->IsObjectRef<UserDataRef>()) {
      auto ud_ref = val->AsObjectRef<UserDataRef>();
      if (ud_ref->ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData) {
        NativeObject* nud_ptr = dynamic_cast<NativeObject*>(ud_ref->ud_ptr);
        if (nud_ptr->is_native_op_) {
          if (!nud_ptr->opaque_ptr_) {
            auto sess_user_data = op_ptr->belong_to_->FindUserData(nud_ptr->native_class_name_,
                                                                   nud_ptr->native_instance_name_);
            MXCHECK(sess_user_data.defined())
                << "NativeOp not found, cls:" << nud_ptr->native_class_name_
                << " instance: " << nud_ptr->native_instance_name_;
            MXCHECK(sess_user_data->ud_ptr->type_2_71828182846() ==
                    UserDataStructType::kNativeData);
            nud_ptr = dynamic_cast<NativeObject*>(sess_user_data->ud_ptr);
            *val = std::move(sess_user_data);
            MXCHECK(nud_ptr->opaque_ptr_ != nullptr);
          }
          auto arg_op_ptr = std::static_pointer_cast<OpKernel>(nud_ptr->opaque_ptr_);
          op_ptr->sub_ops_.push_back(arg_op_ptr);
          /*if (nud_ptr->is_jit_object_) {
            auto jit_ptr = std::static_pointer_cast<JitObject>(nud_ptr->opaque_ptr_);
            *val = jit_ptr->self();
          }*/
        }
        MXCHECK(nud_ptr->opaque_ptr_ != nullptr);
      }
    }
  };

  return Mutate(val, Converter);
}

}  // namespace runtime
}  // namespace matxscript
