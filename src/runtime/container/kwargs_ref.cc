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
#include <matxscript/runtime/container/kwargs_ref.h>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/kwargs_private.h>
#include <matxscript/runtime/memory.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Kwargs container
 *****************************************************************************/

Kwargs::Kwargs() {
  auto n = make_object<KwargsNode>();
  data_ = std::move(n);
}

Kwargs::Kwargs(std::initializer_list<value_type> init) {  // NOLINT(*)
  data_ = make_object<KwargsNode>(init.begin(), init.end());
}

RTValue& Kwargs::get_item(string_view key) const {
  MX_CHECK_DPTR(Kwargs);
  auto iter = d->data_container.find(key);
  MXCHECK(iter != d->data_container.end()) << "Kwargs[" << key << "] not found";
  return iter->second;
}

RTValue& Kwargs::operator[](string_view key) const {
  MX_CHECK_DPTR(Kwargs);
  return d->data_container[std::move(key)];
}

int64_t Kwargs::size() const {
  MX_DPTR(Kwargs);
  return d == nullptr ? 0 : d->data_container.size();
}

bool Kwargs::empty() const {
  return size() == 0;
}

bool Kwargs::contains(const string_view& key) const {
  MX_DPTR(Kwargs);
  return d == nullptr ? false : d->data_container.find(key) != d->data_container.end();
}

string_view Kwargs::diff(string_view* args, size_t num_args) const {
  MX_DPTR(Kwargs);
  auto b = d->data_container.begin();
  auto e = d->data_container.end();
  auto args_last = args + num_args;
  for (; b != e; ++b) {
    if (args_last == std::find(args, args_last, b->first)) {
      return b->first;
    }
  }
  return {};
}

void KwargsUnpackHelper::unpack(RTView* pos_args, PyArgs original_args) const {
  int64_t default_begin_pos = num_args_ - num_default_args_;
  int64_t num_original_args = original_args.size();
  num_original_args -= 1;
  Kwargs kwargs = original_args[num_original_args].AsNoCheck<Kwargs>();
  for (auto i = 0; i < num_original_args; ++i) {
    if (kwargs.contains(arg_names_[i])) {
      THROW_PY_TypeError(func_name_, "() got multiple values for argument '", arg_names_[i], "'");
    }
    pos_args[i] = original_args[i].As<RTView>();
  }
  int num_checker = kwargs.size();
  for (auto i = num_original_args; i < num_args_; ++i) {
    if (kwargs.contains(arg_names_[i])) {
      pos_args[i] = kwargs[arg_names_[i]].template As<RTView>();
      --num_checker;
    } else if (i >= default_begin_pos) {
      pos_args[i] = default_args_[i - default_begin_pos].As<RTView>();
    } else {
      THROW_PY_TypeError(
          func_name_, "() missing 1 required positional argument: '", arg_names_[i], "'");
    }
  }
  if (num_checker != 0) {
    THROW_PY_TypeError(func_name_,
                       "() got an unexpected keyword argument '",
                       kwargs.diff(arg_names_, num_args_),
                       "'");
  }
}

void KwargsUnpackHelper::unpack(RTView* pos_args,
                                MATXScriptAny* original_args,
                                int num_original_args) const {
  int64_t default_begin_pos = num_args_ - num_default_args_;
  num_original_args -= 1;
  Kwargs kwargs = RTView(original_args[num_original_args]).AsNoCheck<Kwargs>();
  for (auto i = 0; i < num_original_args; ++i) {
    if (kwargs.contains(arg_names_[i])) {
      THROW_PY_TypeError(func_name_, "() got multiple values for argument '", arg_names_[i], "'");
    }
    pos_args[i] = RTView(original_args[i]);
  }
  int num_checker = kwargs.size();
  for (auto i = num_original_args; i < num_args_; ++i) {
    if (kwargs.contains(arg_names_[i])) {
      pos_args[i] = kwargs[arg_names_[i]].template As<RTView>();
      --num_checker;
    } else if (i >= default_begin_pos) {
      pos_args[i] = default_args_[i - default_begin_pos].As<RTView>();
    } else {
      THROW_PY_TypeError(
          func_name_, "() missing 1 required positional argument: '", arg_names_[i], "'");
    }
  }
  if (num_checker != 0) {
    THROW_PY_TypeError(func_name_,
                       "() got an unexpected keyword argument '",
                       kwargs.diff(arg_names_, num_args_),
                       "'");
  }
}

template <>
bool IsConvertible<Kwargs>(const Object* node) {
  return node ? node->IsInstance<Kwargs::ContainerType>() : Kwargs::_type_is_nullable;
}

std::ostream& operator<<(std::ostream& os, Kwargs const& n) {
  auto* kw_node = static_cast<const KwargsNode*>(n.get());
  os << '{';
  for (auto it = kw_node->begin(); it != kw_node->end(); ++it) {
    if (it != kw_node->begin()) {
      os << ", ";
    }
    os << it->first << ": ";
    if (it->second.IsString()) {
      os << "b'" << it->second.As<string_view>() << "'";
    } else if (it->second.IsUnicode()) {
      os << "\'" << it->second.As<unicode_view>() << "\'";
    } else {
      os << it->second;
    }
  }
  os << '}';
  return os;
}

}  // namespace runtime
}  // namespace matxscript
