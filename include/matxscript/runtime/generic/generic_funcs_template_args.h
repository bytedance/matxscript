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

#include "./generic_funcs.h"

/******************************************************************************
 * This file is only for cc_test
 *****************************************************************************/

namespace matxscript {
namespace runtime {

/******************************************************************************
 * builtin object's member function
 *
 * Function schema :
 *    RTValue unbound_function(self, *args);
 *
 *****************************************************************************/

#define MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RetType, FuncName)                                 \
  template <typename... Args>                                                                    \
  static inline RetType FuncName(RTView self, Args&&... args) {                                  \
    return FuncName(static_cast<const Any&>(self),                                               \
                    PyArgs{std::initializer_list<RTView>{RTView(std::forward<Args>(args))...}}); \
  }

MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_append);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_add);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_extend);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_clear);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_reserve);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_capacity);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_bucket_count);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_find);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_update);

// str/bytes/regex
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_lower);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_upper);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_isdigit);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_isalpha);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_encode);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_decode);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_split);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_join);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_replace);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_match);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_startswith);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_endswith);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_lstrip);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_rstrip);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_strip);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_count);

// dict
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_keys);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_values);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_items);
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_get);

// NDArray
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_to_list);

// trie tree
MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_prefix_search);

#undef MATXSCRIPT_KERNEL_OBJECT_UNBOUND_FUNC

/******************************************************************************
 * python simple builtin modules and functions
 *
 * Function schema:
 *     RTValue module_method(*args);
 *
 *****************************************************************************/

#define MATXSCRIPT_KERNEL_GLOBAL_FUNC(RetType, FuncName)                                         \
  template <typename... Args>                                                                    \
  static inline RetType FuncName(Args&&... args) {                                               \
    return FuncName(PyArgs{std::initializer_list<RTView>{RTView(std::forward<Args>(args))...}}); \
  }

// json
MATXSCRIPT_KERNEL_GLOBAL_FUNC(RTValue, kernel_json_load);
MATXSCRIPT_KERNEL_GLOBAL_FUNC(RTValue, kernel_json_loads);
MATXSCRIPT_KERNEL_GLOBAL_FUNC(Unicode, kernel_json_dumps);

// file
MATXSCRIPT_KERNEL_GLOBAL_FUNC(File, kernel_file_open);

// builtin math func
MATXSCRIPT_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_min);
MATXSCRIPT_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_max);
MATXSCRIPT_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_and);
MATXSCRIPT_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_or);

}  // namespace runtime
}  // namespace matxscript
