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
#include <matxscript/pipeline/jit_object.h>

#include <matxscript/pipeline/tx_session.h>
#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/container/unicode_helper.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/cxxabi_helper.h>
#include <matxscript/runtime/file_util.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>
#include <tuple>
#include <utility>

#include "userdata_mutator.h"

namespace matxscript {
namespace runtime {

MATX_REGISTER_NATIVE_OP(JitObject);

JitObjectPtr check_get_jit_object(const UserDataRef& ud) {
  MXCHECK(ud->ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData);
  auto nud_ptr = dynamic_cast<NativeObject*>(ud->ud_ptr);
  MXCHECK(nud_ptr && nud_ptr->is_jit_object_);
  auto obj_ptr =
      std::static_pointer_cast<JitObject>(std::static_pointer_cast<OpKernel>(nud_ptr->opaque_ptr_));
  return obj_ptr;
}

JitObjectPtr try_get_jit_object(const UserDataRef& ud) {
  if (ud.defined() && ud->ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData) {
    auto nud_ptr = dynamic_cast<NativeObject*>(ud->ud_ptr);
    if (nud_ptr && nud_ptr->is_jit_object_) {
      auto obj_ptr = std::static_pointer_cast<JitObject>(
          std::static_pointer_cast<OpKernel>(nud_ptr->opaque_ptr_));
      return obj_ptr;
    }
  }
  return nullptr;
}

// Function schema

JitObject::FuncParam JitObject::FuncParam::FromDict(const Dict& config) {
  String name = config.get_item("name").As<String>();
  int32_t type_code = config.get_item("type_code").As<int32_t>();
  return FuncParam(name, type_code);
}

Dict JitObject::FuncParam::ToDict() const {
  return {{"name", name}, {"type_code", type_code}};
}

JitObject::FuncMeta JitObject::FuncMeta::FromDict(const Dict& config) {
  String name = config.get_item("name").As<String>();
  bool bound_self = config.get_item("bound_self").As<bool>();
  auto generic_args = config.get_item("args").AsObjectRef<List>();
  std::vector<FuncParam> args;
  for (auto& arg : generic_args) {
    args.push_back(FuncParam::FromDict(arg.As<Dict>()));
  }
  List defaults;
  if (config.contains("defaults")) {
    defaults = config.get_item("defaults").AsObjectRef<List>();
  }
  return FuncMeta(std::move(name), bound_self, std::move(args), std::move(defaults));
}

Dict JitObject::FuncMeta::ToDict() const {
  Dict generic_fm;
  generic_fm["name"] = name;
  generic_fm["bound_self"] = bound_self;
  List generic_args;
  for (auto& arg : args) {
    generic_args.push_back(arg.ToDict());
  }
  generic_fm["args"] = std::move(generic_args);
  generic_fm["defaults"] = defaults;
  return generic_fm;
}

JitObject::ClassMeta JitObject::ClassMeta::FromDict(const Dict& config) {
  // name and slot
  String name = config.get_item("name").As<String>();
  int32_t len_slots = config.get_item("len_slots").As<int32_t>();

  // init function
  FuncMeta init_func;
  if (config.contains("init_func")) {
    init_func = FuncMeta::FromDict(config["init_func"].As<Dict>());
  } else {
    MXLOG(INFO) << "class has no __init__ function";
  }

  // __init__ arguments
  std::vector<RTValue> init_args;
  if (config.contains("init_args")) {
    auto generic_init_args = config.get_item("init_args").AsObjectRef<List>();
    for (size_t i = 0; i < generic_init_args.size(); ++i) {
      init_args.push_back(generic_init_args[i]);
    }
  }
  // member functions
  std::vector<FuncMeta> member_funcs;
  if (config.contains("member_funcs")) {
    for (const auto& js_fm : config["member_funcs"].AsObjectRef<List>()) {
      member_funcs.push_back(FuncMeta::FromDict(js_fm.As<Dict>()));
    }
  }
  return ClassMeta(std::move(name),
                   len_slots,
                   std::move(init_func),
                   std::move(init_args),
                   std::move(member_funcs));
}

Dict JitObject::ClassMeta::ToDict() const {
  Dict generic_class_meta;

  // name and slot
  generic_class_meta["name"] = name;
  generic_class_meta["len_slots"] = len_slots;

  // constructor
  generic_class_meta["init_func"] = init_func.ToDict();

  // init args
  List generic_init_args;
  for (auto& init_arg : init_args) {
    generic_init_args.push_back(init_arg);
  }
  generic_class_meta["init_args"] = std::move(generic_init_args);

  // member functions
  List generic_mem_funcs;
  for (auto& fm : member_funcs) {
    generic_mem_funcs.push_back(fm.ToDict());
  }
  generic_class_meta["member_funcs"] = std::move(generic_mem_funcs);

  return generic_class_meta;
}

// JitObject::Options
JitObject::Options JitObject::Options::FromDict(const Dict& config) {
  JitObject::Options jit_module_opts;
  // thread safe
  if (config.contains("share")) {
    jit_module_opts.share = config.get_item("share").As<bool>();
  }
  // base info
  jit_module_opts.dso_path = config.get_item("dso_path").As<String>();
  jit_module_opts.dso_path_cxx11 = config.get_item("dso_path_cxx11").As<String>();

  // bundle info
  if (config.contains("need_bundle")) {
    for (auto& arg_name : config["need_bundle"].AsObjectRef<List>()) {
      jit_module_opts.need_bundle.push_back(arg_name.As<String>());
    }
  }

  // captures
  if (config.contains("captures")) {
    for (auto& cls_and_name : config["captures"].AsObjectRef<List>()) {
      auto tup = cls_and_name.As<Tuple>();
      MXCHECK(tup.size() == 2);
      jit_module_opts.captures.push_back(std::make_pair(tup[0].As<String>(), tup[1].As<String>()));
    }
  }

  // class or function
  if (config.contains("class_info")) {
    jit_module_opts.is_class = true;
    jit_module_opts.class_info = ClassMeta::FromDict(config["class_info"].As<Dict>());
  } else {
    jit_module_opts.is_class = false;
    jit_module_opts.func_info = FuncMeta::FromDict(config["func_info"].As<Dict>());
  }
  return jit_module_opts;
}

Dict JitObject::Options::ToDict() const {
  Dict generic_object_opt;

  // share
  generic_object_opt["share"] = share;

  // base info
  generic_object_opt["dso_path"] = dso_path;
  generic_object_opt["dso_path_cxx11"] = dso_path_cxx11;

  // bundle resource
  List generic_bundle;
  for (const auto& arg_name : need_bundle) {
    generic_bundle.push_back(arg_name);
  }
  generic_object_opt["need_bundle"] = std::move(generic_bundle);

  // captures
  List generic_captures;
  for (const auto& cls_and_name : captures) {
    generic_captures.push_back(Tuple::dynamic(cls_and_name.first, cls_and_name.second));
  }
  generic_object_opt["captures"] = std::move(generic_captures);

  if (is_class) {
    generic_object_opt["class_info"] = class_info.ToDict();
  } else {
    generic_object_opt["func_info"] = func_info.ToDict();
  }

  return generic_object_opt;
}

std::pair<int, string_view::size_type> JsonPathGetter_NextToken(unicode_view jsonpath) {
  /*
    returns:
      (TokenType, TokenLen):
        TokenType: -1 for eos, 0 for list index, 1 for dict key
  */
  if (jsonpath.length() == 0) {
    return std::make_pair(-1, string_view::npos);
  }
  if (jsonpath[0] == U'[') {
    string_view::size_type token_len = jsonpath.find(U']') + 1;
    return std::make_pair(0, token_len);
  } else {
    string_view::size_type token_len = std::min(jsonpath.find_first_of(U".["), jsonpath.length());
    return std::make_pair(1, token_len);
  }
}

RTValue* JsonPathGetter(RTValue& obj, unicode_view jsonpath) {
  const RTValue* p = &obj;
  string_view::size_type b = 0;
  std::pair<int, string_view::size_type> token = JsonPathGetter_NextToken(jsonpath);
  while (token.first != -1) {
    unicode_view token_s = jsonpath.substr(b, token.second);
    if (token.first == 0) {
      List::size_type idx = Kernel_int64_t::make(Unicode(token_s.substr(1, token_s.length() - 2)));
      p = &(p->AsObjectView<List>().data().get_item(idx));
    } else if (token.first == 1) {
      p = &(p->AsObjectView<Dict>().data().get_item(token_s));
    }
    b += token.second;
    while (jsonpath[b] == U'.') {
      ++b;
    }
    token = JsonPathGetter_NextToken(jsonpath.substr(b));
  }
  return const_cast<RTValue*>(p);
}

int JitObject::Bundle(string_view folder) {
  Options new_opt = options_;
  String src_path_cxx11 = options_.dso_path_cxx11 + ".enc_cc";
  String src_dso_path = options_.dso_path + ".enc_cc";
  if (FileUtil::Exists(src_path_cxx11)) {
    BundlePath(src_path_cxx11, folder);
  }
  if (FileUtil::Exists(src_dso_path)) {
    BundlePath(src_dso_path, folder);
  }
  new_opt.dso_path_cxx11 = BundlePath(new_opt.dso_path_cxx11, folder);
  new_opt.dso_path = BundlePath(new_opt.dso_path, folder);
  if (new_opt.is_class) {
    for (string_view jsonpath : new_opt.need_bundle) {
      if (jsonpath[0] == '$' && jsonpath[1] == '.') {
        string_view::size_type b = 2;
        string_view::size_type e = std::min(jsonpath.find_first_of(".[", b), jsonpath.length());
        string_view var_name = jsonpath.substr(b, e - b);
        int64_t arg_index = -1;
        for (size_t i = 0; i < new_opt.class_info.init_func.args.size(); ++i) {
          if (new_opt.class_info.init_func.args[i].name.view() == var_name) {
            arg_index = i;
            break;
          }
        }
        MXCHECK_GE(arg_index, 0) << "var name not found when bundling: " << var_name;
        while (jsonpath[e] == U'.') {
          ++e;
        }
        RTValue* path = JsonPathGetter(new_opt.class_info.init_args[arg_index],
                                       StringHelper::Decode(jsonpath.substr(e)));
        String path_s = UnicodeHelper::Encode(path->As<unicode_view>());
        path_s = BundlePath(path_s, folder);
        *path = StringHelper::Decode(path_s);
      } else {
        // for backwards compatibility
        string_view var_name = jsonpath;
        int64_t arg_index = -1;
        for (size_t i = 0; i < new_opt.class_info.init_func.args.size(); ++i) {
          if (new_opt.class_info.init_func.args[i].name.view() == var_name) {
            arg_index = i;
            break;
          }
        }
        MXCHECK_GE(arg_index, 0) << "var name not found when bundling: " << var_name;
        Unicode path = new_opt.class_info.init_args[arg_index].As<Unicode>();
        String path_s = path.encode();
        path_s = BundlePath(path_s, folder);
        new_opt.class_info.init_args[arg_index] = String(path_s).decode();
      }
    }
  }
  attributes_ = Attributes::FromDict(new_opt.ToDict());
  return 0;
}

JitObject::NativeMethod JitObject::MakeNativeFunc(const FuncMeta& meta,
                                                  UserDataRef self,
                                                  MATXScriptBackendPackedCFunc c_packed_func) {
  if (meta.bound_self) {
    return [self, c_packed_func](void* jit_obj, PyArgs args) -> RTValue {
      const int kNumArgs = args.size() + 1;
      const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
      MATXScriptAny values[kArraySize];
      // always bound self
      values[0] = RTView(self).value();
      for (size_t i = 0; i < args.size(); ++i) {
        values[i + 1] = args[i].value();
      }
      MATXScriptAny out_ret_value;
      c_packed_func(values, kNumArgs, &out_ret_value, self->ud_ptr);
      return RTValue::MoveFromCHost(&out_ret_value);
    };
  } else {
    return [self, c_packed_func](void* jit_obj, PyArgs args) -> RTValue {
      const int kNumArgs = args.size();
      const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
      MATXScriptAny values[kArraySize];
      // no need bound self
      for (size_t i = 0; i < args.size(); ++i) {
        values[i] = args[i].value();
      }
      MATXScriptAny out_ret_value;
      c_packed_func(values, kNumArgs, &out_ret_value, self->ud_ptr);
      return RTValue::MoveFromCHost(&out_ret_value);
    };
  }
}

void JitObject::Init() {
  options_ = Options::FromDict(attributes_.ToDict());
  if (options_.is_class) {
    char seed[64];
    snprintf(seed,
             sizeof(seed),
             "%d_%d_%d",
             options_.class_info.len_slots,
             (int)options_.class_info.init_args.size(),
             (int)options_.class_info.member_funcs.size());
    name_ = GlobalUniqueIndex::instance()->gen_uniq_name(options_.class_info.name, seed);
  } else {
    char seed[64];
    snprintf(seed, sizeof(seed), "%d", (int)options_.func_info.args.size());
    name_ = GlobalUniqueIndex::instance()->gen_uniq_name(options_.func_info.name, seed);
  }
  String dso_path = options_.dso_path;
#ifdef MATX_SUPPORT_ANDROID
  dso_path = options_.dso_path_cxx11;
#else
  if (MATXSCRIPT_FLAGS_GLIBCXX_USE_CXX11_ABI) {
    dso_path = options_.dso_path_cxx11;
  }
#endif
  MXCHECK(belong_to_ != nullptr) << "belong_to_ is not set";
  auto abs_dso_path = resource_path_ + dso_path;
  if (MATXSCRIPT_FLAGS_GLIBCXX_USE_CXX11_ABI) {
    MXCHECK((!dso_path.empty()) && FileUtil::IsRegularFile(abs_dso_path) &&
            FileUtil::Exists(abs_dso_path))
        << "dso path not found: " << abs_dso_path << "\n"
        << "Please check gcc8 was available when tracing a model for c++ server";
  } else {
    MXCHECK(FileUtil::Exists(abs_dso_path)) << "dso path not found: " << abs_dso_path;
  }
  module_ = Module::LoadFromFile(abs_dso_path);
  auto class_init_args = options_.class_info.init_args;
  if (options_.is_class) {
    std::unordered_map<int64_t, std::vector<String>> bundle_args;
    for (string_view jsonpath : options_.need_bundle) {
      if (jsonpath[0] == '$' && jsonpath[1] == '.') {
        string_view::size_type b = 2;
        string_view::size_type e = std::min(jsonpath.find_first_of(".[", b), jsonpath.length());
        string_view var_name = jsonpath.substr(b, e - b);
        int64_t arg_index = -1;
        for (size_t i = 0; i < options_.class_info.init_func.args.size(); ++i) {
          if (options_.class_info.init_func.args[i].name.view() == var_name) {
            arg_index = i;
            break;
          }
        }
        MXCHECK_GE(arg_index, 0) << "var name not found when bundling: " << var_name;
        while (jsonpath[e] == U'.') {
          ++e;
        }
        bundle_args[arg_index].push_back(jsonpath.substr(e));
      } else {
        // for backwards compatibility
        string_view var_name = jsonpath;
        int64_t arg_index = -1;
        for (size_t i = 0; i < options_.class_info.init_func.args.size(); ++i) {
          if (options_.class_info.init_func.args[i].name.view() == var_name) {
            arg_index = i;
            break;
          }
        }
        MXCHECK_GE(arg_index, 0) << "var name not found when bundling: " << var_name;
        bundle_args[arg_index].push_back("");
      }
    }
    for (auto i = 0; i < class_init_args.size(); ++i) {
      RTValue& arg_i = class_init_args[i];
      UserDataMutator::Mutate(&arg_i, this);
      if (bundle_args.count(i)) {
        for (const auto& p : bundle_args[i]) {
          RTValue* path = JsonPathGetter(arg_i, StringHelper::Decode(p));
          *path = resource_path_.decode() + path->As<Unicode>();
        }
      }
    }
  }

  // init function table
  function_table_.clear();

  // constructor self
  if (options_.is_class) {
    auto cons_wrapper = module_.GetFunction(
        FunctionNameRules::add_wrapper_suffix(options_.class_info.init_func.name));
    int32_t num_args = options_.class_info.init_func.args.size();
    for (size_t i = 0; i < num_args; ++i) {
      if (options_.class_info.init_func.args[i].type_code != INT16_MIN) {
        MXCHECK(options_.class_info.init_func.args[i].type_code == class_init_args[i].type_code())
            << "[JitObject::Initialize][call: " << options_.class_info.init_func.name
            << "] Expect argument[" << i << "] type is "
            << TypeIndex2Str(options_.class_info.init_func.args[i].type_code) << " but get "
            << class_init_args[i].type_name();
      }
    }
    // bundle session handler
    class_init_args.push_back(RTValue(belong_to_));
    auto rv = cons_wrapper(PyArgs(class_init_args.data(), class_init_args.size()));
    self_ = rv.MoveToObjectRef<UserDataRef>();
    MXCHECK(self_->ud_ptr) << "UserData ptr invalid";
  } else {
    void* reg = module_.GetFunction(symbol::library_func_registry)({}).As<void*>();
    auto* func_reg = (MATXScriptFuncRegistry*)reg;
    int idx = LookupFuncRegistryName(func_reg->names, options_.func_info.name);
    MXCHECK(idx >= 0) << "[JitObject] function not found, name: " << options_.func_info.name;
    self_ = UserDataRef(
        0,
        0,
        new UserFunction(options_.func_info.name, func_reg->funcs[idx], belong_to_),
        [](ILightUserData* self) -> void { delete reinterpret_cast<UserFunction*>(self); });
  }
  self_->safe_deleter.module = module_;  // safe deleter

  if (options_.is_class) {
    if (self_->ud_ptr->type_2_71828182846() == UserDataStructType::kUserData) {
      IUserDataRoot* ud = static_cast<IUserDataRoot*>(self_->ud_ptr);
      for (auto& func_meta : options_.class_info.member_funcs) {
        auto name_bound =
            FunctionNameRules::remove_class_prefix(options_.class_info.name, func_meta.name);
        if (name_bound == "__init__") {
          continue;
        }
        auto ft_itr = ud->function_table_2_71828182846_->find(name_bound);
        MXCHECK(ft_itr != ud->function_table_2_71828182846_->end())
            << "[Class:" << options_.class_info.name
            << "] member function not found: " << name_bound;
        auto native_func = MakeNativeFunc(func_meta, self_, ft_itr->second);
        function_table_.emplace(name_bound, native_func);
      }
    }
  } else {
    auto self = this->self_;
    auto native_func = [self](void* jit_obj, PyArgs args) -> RTValue {
      return self.generic_call(args);
    };
    function_table_.emplace(options_.func_info.name.view(), native_func);
  }

  // captures
  for (auto& cls_and_name : options_.captures) {
    auto ud = belong_to_->FindUserData(cls_and_name.first, cls_and_name.second);
    MXCHECK(ud.defined());
    sub_ops_.push_back(check_get_op_kernel(ud));
  }
}

std::pair<NativeFunction, const JitObject::FuncMeta*> JitObject::GetFunction(
    string_view name_view) {
  String name(name_view.data(), name_view.size());
  if (options_.is_class) {
    for (auto& member_func : options_.class_info.member_funcs) {
      if (member_func.name == name) {
        return std::make_pair(module_->GetFunction(name), &member_func);
      }
    }
    return std::make_pair(module_->GetFunction(name), nullptr);
  } else {
    return std::make_pair(module_->GetFunction(name), &options_.func_info);
  }
}

const String& JitObject::PyObjectName() const {
  if (options_.is_class) {
    return options_.class_info.name;
  } else {
    return options_.func_info.name;
  }
}

RTValue JitObject::generic_call_attr(string_view func_name, PyArgs args) {
  return self_.generic_call_attr(func_name, args);
}

}  // namespace runtime
}  // namespace matxscript
