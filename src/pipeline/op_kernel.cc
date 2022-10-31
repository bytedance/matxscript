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
#include <matxscript/pipeline/op_kernel.h>

#include <matxscript/pipeline/jit_object.h>
#include <matxscript/pipeline/jit_op.h>
#include <matxscript/pipeline/pickle.h>
#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/file_util.h>
#include <matxscript/runtime/native_object_registry.h>

namespace matxscript {
namespace runtime {

static String GenRelativeBundlePath(const OpKernel* op, string_view prefix) {
  String relative_dir;
  String op_name = op->GetName();
  string_view cls_name = op->ClassName();
  if (op_name.startswith(cls_name)) {
    relative_dir = op_name;
  } else {
    relative_dir = String(cls_name) + "_" + op_name;
  }
  return relative_dir;
}

String OpKernel::BundlePath(string_view location, string_view folder) const {
  if (location.empty()) {
    return String();
  }
  String relative_dir = GenRelativeBundlePath(this, folder);
  String dst = String(folder) + "/" + relative_dir + "/";
  FileUtil::Mkdir(dst);
  FileUtil::Copy((resource_path_ + location).c_str(), dst.c_str());
  return relative_dir + "/" + FileUtil::BaseName(location);
}

void OpKernel::Initialize(Attributes attrs) {
  if (attrs.HasAttr(PREFIX_KEY)) {
    resource_path_ = attrs.GetAttr<String>(PREFIX_KEY);
  }
  if (belong_to_ && belong_to_->GetDevice() != NONE_DEVICE) {
    device_ = belong_to_->GetDevice();
  }
  auto json_config = pickle::ToJsonStruct(RTValue(attrs.ToDict()));
  auto config = JsonUtil::ToString(&json_config);
  name_ = GlobalUniqueIndex::instance()->gen_uniq_name(class_name_, config);
  attributes_ = std::move(attrs);
  sub_ops_.clear();
  this->Init();
}

void OpKernel::SetBelongTo(TXSession* belong_to) {
  belong_to_ = belong_to;
}

OpKernelPtr OpKernel::GetOpImpl(string_view cls, string_view name) {
  auto ptr = belong_to_ ? belong_to_->FindOp(cls, name) : nullptr;
  if (ptr) {
    sub_ops_.push_back(ptr);
  }
  return ptr;
}

OpKernelPtr check_get_op_kernel(const UserDataRef& ud) {
  MXCHECK(ud->ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData);
  auto nud_ptr = dynamic_cast<NativeObject*>(ud->ud_ptr);
  MXCHECK(nud_ptr && nud_ptr->is_native_op_);
  auto op_ptr = std::static_pointer_cast<OpKernel>(nud_ptr->opaque_ptr_);
  return op_ptr;
}

OpKernelPtr try_get_op_kernel(const UserDataRef& ud) {
  if (ud.defined() && ud->ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData) {
    auto nud_ptr = dynamic_cast<NativeObject*>(ud->ud_ptr);
    if (nud_ptr && nud_ptr->is_native_op_) {
      auto op_ptr = std::static_pointer_cast<OpKernel>(nud_ptr->opaque_ptr_);
      return op_ptr;
    }
  }
  return nullptr;
}

UserDataRef make_userdata(OpKernelPtr op_ptr) {
  auto native_op_register = NativeObjectRegistry::Get(op_ptr->ClassName());
  MXCHECK(native_op_register != nullptr) << "Native OP not found: " << op_ptr->ClassName();
  std::shared_ptr<void> opaque_ptr = op_ptr;
  NativeObject* ud = new NativeObject(opaque_ptr);
  ud->is_native_op_ = native_op_register->is_native_op_;
  ud->is_jit_object_ = native_op_register->is_jit_object_;
  ud->function_table_ = &native_op_register->function_table_;
  ud->native_class_name_ = native_op_register->class_name;
  ud->native_instance_name_ = op_ptr->GetName();
  if (ud->is_jit_object_) {
    auto* jit_ptr = static_cast<JitObject*>(op_ptr.get());
    ud->function_table_ = jit_ptr->GetFunctionTable();
  }
  return UserDataRef(
      ud->tag_2_71828182846(), ud->size_2_71828182846(), ud, default_userdata_deleter);
}

UserDataRef make_op_kernel(string_view class_name, PyArgs args, TXSession* sess) {
  auto native_op_register = NativeObjectRegistry::Get(class_name);
  MXCHECK(native_op_register != nullptr) << "Native OP not found: " << class_name;
  MXCHECK(native_op_register->is_native_op_) << class_name << " is not Native OP";
  auto opaque_ptr = native_op_register->construct(args);
  NativeObject* ud = new NativeObject(opaque_ptr);
  ud->is_native_op_ = native_op_register->is_native_op_;
  ud->is_jit_object_ = native_op_register->is_jit_object_;
  ud->function_table_ = &native_op_register->function_table_;
  ud->native_class_name_ = native_op_register->class_name;
  auto op_ptr = std::static_pointer_cast<OpKernel>(opaque_ptr);
  op_ptr->class_name_ = native_op_register->class_name;
  auto attrs = ::matxscript::runtime::Attributes::FromDict(args[0].As<Dict>());
  op_ptr->SetBelongTo(sess);
  op_ptr->Initialize(std::move(attrs));
  ud->native_instance_name_ = op_ptr->GetName();
  if (ud->is_jit_object_) {
    auto* jit_ptr = static_cast<JitObject*>(op_ptr.get());
    ud->function_table_ = jit_ptr->GetFunctionTable();
  }
  return UserDataRef(
      ud->tag_2_71828182846(), ud->size_2_71828182846(), ud, default_userdata_deleter);
}

// example
namespace {

class TableLookupExampleOp : public OpKernel {
 public:
  String vocab_file;

 public:
  void Init() override {
    vocab_file = GetAttr<Unicode>("vocab_file").encode();
    FileReader reader(vocab_file);
    const char* line;
    size_t len = 0;
    int64_t idx = 0;
    while (reader.ReadLine(&line, &len)) {
      term2id_.emplace(String(line, len).decode(), idx);
      ++idx;
    }
  }

  int Bundle(string_view folder) override {
    auto new_loc = BundlePath(vocab_file, folder);
    SetAttr("vocab_file", std::move(new_loc));
    return 0;
  }

  RTValue Process(PyArgs inputs) const override {
    CheckArgs(inputs.size(), 1);
    auto& input = inputs[0];
    switch (input.type_code()) {
      case TypeIndex::kRuntimeString: {
        return Process(input.As<string_view>());
      } break;
      case TypeIndex::kRuntimeUnicode: {
        return Process(input.As<unicode_view>());
      } break;
      case TypeIndex::kRuntimeList: {
        return Process(input.AsObjectViewNoCheck<List>().data());
      } break;
      default: {
        /* not compatible type */
        MXCHECK(false) << "input type error, \n"
                       << "optional: List[str] or str, \n"
                       << "but receive type : " << input.type_name();
      }
    }
    return None;
  }

  List Process(const List& input) const {
    List output;
    output.reserve(input.size());
    for (auto& item : input) {
      switch (item.type_code()) {
        case TypeIndex::kRuntimeList: {
          auto rsl = Process(item.AsObjectViewNoCheck<List>().data());
          output.push_back(std::move(rsl));
        } break;
        case TypeIndex::kRuntimeString: {
          output.push_back(Process(item.As<string_view>()));
        } break;
        case TypeIndex::kRuntimeUnicode: {
          output.push_back(Process(item.As<unicode_view>()));
        } break;
        default: {
          MXCHECK(false) << "[RegexSplitOp] unsupported data type: " << item.type_name();
        } break;
      }
    }
    return output;
  }

  int64_t Process(string_view input) const {
    return Process(StringHelper::Decode(input));
  }

  int64_t Process(unicode_view input) const {
    auto itr = term2id_.find(Unicode(input));
    return itr == term2id_.end() ? -1 : itr->second;
  }

 private:
  ska::flat_hash_map<Unicode, int64_t> term2id_;
};

MATX_REGISTER_NATIVE_OP(TableLookupExampleOp);

}  // namespace

}  // namespace runtime
}  // namespace matxscript
