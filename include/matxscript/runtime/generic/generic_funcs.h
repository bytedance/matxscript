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

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/str_escape.h>
#include <cstdint>

namespace matxscript {
namespace runtime {

// Compared to the virtual function implemented in an object,
// the distribution here will not increase the object bytes

/******************************************************************************
 * User custom object dispatch
 *****************************************************************************/

RTValue kernel_object___dispatch__(const Any& self, string_view func_name, PyArgs args);

/******************************************************************************
 * python object data model special method names
 *
 * url: https://docs.python.org/3/reference/datamodel.html#special-method-names
 * url: https://docs.python.org/3/reference/datamodel.html#emulating-container-types
 *
 * object.__len__(self)
 * object.__getitem__(self, key)
 * object.__setitem__(self, key, value)
 * object.__delitem__(self, key)
 * object.__contains__(self, item)
 * object.__hash__(self)
 * object.__reversed__(self)
 *
 *****************************************************************************/

// Function signature is known
// __len__
int64_t kernel_object___len__(const Any& self);
template <typename SELF_TYPE,
          typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_object___len__(const SELF_TYPE& self) {
  return kernel_object___len__(static_cast<const Any&>(RTView(self)));
}

// __getitem__
RTValue kernel_object___getitem__(const Any& self, const Any& key);
template <
    typename SELF_TYPE,
    typename KEY_TYPE,
    typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE, KEY_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___getitem__(const SELF_TYPE& self,
                                                           const KEY_TYPE& key) {
  return kernel_object___getitem__(static_cast<const Any&>(RTView(self)),
                                   static_cast<const Any&>(RTView(key)));
}

// __setitem__
RTValue kernel_object___setitem__(const Any& self, const Any& key, const Any& item);
template <typename SELF_TYPE,
          typename KEY_TYPE,
          typename ITEM_TYPE,
          typename = typename std::enable_if<
              !all_is_runtime_value<SELF_TYPE, KEY_TYPE, ITEM_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___setitem__(const SELF_TYPE& self,
                                                           const KEY_TYPE& key,
                                                           const ITEM_TYPE& item) {
  return kernel_object___setitem__(static_cast<const Any&>(RTView(self)),
                                   static_cast<const Any&>(RTView(key)),
                                   static_cast<const Any&>(RTView(item)));
}

// __delitem__
RTValue kernel_object___delitem__(const Any& self, const Any& key);
template <
    typename SELF_TYPE,
    typename KEY_TYPE,
    typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE, KEY_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___delitem__(const SELF_TYPE& self,
                                                           const KEY_TYPE& item) {
  return kernel_object___delitem__(static_cast<const Any&>(RTView(self)),
                                   static_cast<const Any&>(RTView(item)));
}

// __getslice__
RTValue kernel_object___getslice__(const Any& self,
                                   const Any& start,
                                   const Any& end,
                                   const Any& step);
template <typename SELF_TYPE,
          typename START_TYPE,
          typename END_TYPE,
          typename STEP_TYPE,
          typename = typename std::enable_if<
              !all_is_runtime_value<SELF_TYPE, START_TYPE, END_TYPE, STEP_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___getslice__(const SELF_TYPE& self,
                                                            const START_TYPE& start,
                                                            const END_TYPE& end,
                                                            const STEP_TYPE& step) {
  return kernel_object___getslice__(static_cast<const Any&>(RTView(self)),
                                    static_cast<const Any&>(RTView(start)),
                                    static_cast<const Any&>(RTView(end)),
                                    static_cast<const Any&>(RTView(step)));
}

// __setslice__
RTValue kernel_object___setslice__(const Any& self,
                                   const Any& start,
                                   const Any& end,
                                   const Any& item);
template <typename SELF_TYPE,
          typename START_TYPE,
          typename END_TYPE,
          typename ITEM_TYPE,
          typename = typename std::enable_if<
              !all_is_runtime_value<SELF_TYPE, START_TYPE, END_TYPE, ITEM_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___setslice__(const SELF_TYPE& self,
                                                            const START_TYPE& start,
                                                            const END_TYPE& end,
                                                            const ITEM_TYPE& item) {
  return kernel_object___setslice__(static_cast<const Any&>(RTView(self)),
                                    static_cast<const Any&>(RTView(start)),
                                    static_cast<const Any&>(RTView(end)),
                                    static_cast<const Any&>(RTView(item)));
}

// __reversed__
RTValue kernel_object___reversed__(const Any& self);
template <typename SELF_TYPE,
          typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___reversed__(const SELF_TYPE& self) {
  return kernel_object___reversed__(static_cast<const Any&>(RTView(self)));
}

// __contains__
bool kernel_object___contains__(const Any& self, const Any& item);
template <
    typename SELF_TYPE,
    typename ITEM_TYPE,
    typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE, ITEM_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE bool kernel_object___contains__(const SELF_TYPE& self,
                                                         const ITEM_TYPE& item) {
  return kernel_object___contains__(static_cast<const Any&>(RTView(self)),
                                    static_cast<const Any&>(RTView(item)));
}

// __hash__
RTValue kernel_object___hash__(const Any& self);
template <typename SELF_TYPE,
          typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___hash__(const SELF_TYPE& self) {
  return kernel_object___hash__(static_cast<const Any&>(RTView(self)));
}

// __getattr__
RTValue kernel_object___getattr__(const Any& self, string_view attr);
template <typename SELF_TYPE,
          typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___getattr__(const SELF_TYPE& self,
                                                           string_view attr) {
  return kernel_object___getattr__(static_cast<const Any&>(RTView(self)), attr);
}

// __setattr__
RTValue kernel_object___setattr__(const Any& self, string_view attr, const Any& item);
template <
    typename SELF_TYPE,
    typename ITEM_TYPE,
    typename = typename std::enable_if<!all_is_runtime_value<SELF_TYPE, ITEM_TYPE>::value>::type>
MATXSCRIPT_ALWAYS_INLINE RTValue kernel_object___setattr__(const SELF_TYPE& self,
                                                           string_view attr,
                                                           const ITEM_TYPE& item) {
  return kernel_object___setattr__(
      static_cast<const Any&>(RTView(self)), attr, static_cast<const Any&>(RTView(item)));
}

/******************************************************************************
 * fused builtin object's special method
 *****************************************************************************/

// __fused_getitem__
RTValue kernel_object___fused_getitem__(const Any& self, const PyArgs& keys);

// __fused_setitem__
RTValue kernel_object___fused_setitem__(const Any& self, const PyArgs& keys, const Any& item);

/******************************************************************************
 * builtin object's member function
 *
 * Function schema :
 *    RTValue unbound_function(self, *args);
 *
 *****************************************************************************/

// generic
RTValue kernel_object_append(const Any& self, PyArgs args);
RTValue kernel_object_add(const Any& self, PyArgs args);
RTValue kernel_object_extend(const Any& self, PyArgs args);
RTValue kernel_object_clear(const Any& self, PyArgs args);
RTValue kernel_object_reserve(const Any& self, PyArgs args);
RTValue kernel_object_capacity(const Any& self, PyArgs args);
RTValue kernel_object_bucket_count(const Any& self, PyArgs args);
RTValue kernel_object_find(const Any& self, PyArgs args);
RTValue kernel_object_update(const Any& self, PyArgs args);

// str/bytes/regex
RTValue kernel_object_lower(const Any& self, PyArgs args);
RTValue kernel_object_upper(const Any& self, PyArgs args);
RTValue kernel_object_isdigit(const Any& self, PyArgs args);
RTValue kernel_object_isalpha(const Any& self, PyArgs args);
RTValue kernel_object_encode(const Any& self, PyArgs args);
RTValue kernel_object_decode(const Any& self, PyArgs args);
RTValue kernel_object_split(const Any& self, PyArgs args);
RTValue kernel_object_join(const Any& self, PyArgs args);
RTValue kernel_object_replace(const Any& self, PyArgs args);
RTValue kernel_object_match(const Any& self, PyArgs args);
RTValue kernel_object_startswith(const Any& self, PyArgs args);
RTValue kernel_object_endswith(const Any& self, PyArgs args);
RTValue kernel_object_lstrip(const Any& self, PyArgs args);
RTValue kernel_object_rstrip(const Any& self, PyArgs args);
RTValue kernel_object_strip(const Any& self, PyArgs args);
RTValue kernel_object_count(const Any& self, PyArgs args);
RTValue kernel_object_format(const Any& self, PyArgs args);

// dict
RTValue kernel_object_keys(const Any& self, PyArgs args);
RTValue kernel_object_values(const Any& self, PyArgs args);
RTValue kernel_object_items(const Any& self, PyArgs args);
RTValue kernel_object_get(const Any& self, PyArgs args);

// list
RTValue kernel_object_pop(const Any& self, PyArgs args);
RTValue kernel_object_insert(const Any& self, PyArgs args);
RTValue kernel_object_remove(const Any& self, PyArgs args);
RTValue kernel_object_reverse(const Any& self, PyArgs args);
RTValue kernel_object_sort(const Any& self, PyArgs args);
RTValue kernel_object_index(const Any& self, PyArgs args);

// set
RTValue kernel_object_difference(const Any& self, PyArgs args);
RTValue kernel_object_difference_update(const Any& self, PyArgs args);
RTValue kernel_object_discard(const Any& self, PyArgs args);
RTValue kernel_object_union(const Any& self, PyArgs args);

// NDArray
RTValue kernel_object_to_list(const Any& self, PyArgs args);
RTValue kernel_object_tolist(const Any& self, PyArgs args);
RTValue kernel_object_is_contiguous(const Any& self, PyArgs args);
RTValue kernel_object_shape(const Any& self, PyArgs args);
RTValue kernel_object_dtype(const Any& self, PyArgs args);
RTValue kernel_object_dim(const Any& self, PyArgs args);
RTValue kernel_object_device(const Any& self, PyArgs args);
RTValue kernel_object_transpose(const Any& self, PyArgs args);
RTValue kernel_object_as_type(const Any& self, PyArgs args);
RTValue kernel_object_contiguous(const Any& self, PyArgs args);
RTValue kernel_object_reshape(const Any& self, PyArgs args);
RTValue kernel_object_squeeze(const Any& self, PyArgs args);
RTValue kernel_object_unsqueeze(const Any& self, PyArgs args);

// trie tree
RTValue kernel_object_prefix_search(const Any& self, PyArgs args);
RTValue kernel_object_prefix_search_all(const Any& self, PyArgs args);
RTValue kernel_object_save(const Any& self, PyArgs args);
RTValue kernel_object_load(const Any& self, PyArgs args);

/******************************************************************************
 * python simple builtin modules and functions
 *
 * Function schema:
 *     RTValue module_method(*args);
 *
 *****************************************************************************/

#ifndef DISABLE_UNICODEDATA
// unicodedata
Unicode kernel_unicodedata_normalize(int32_t form, const unicode_view& s);
Unicode kernel_unicodedata_normalize(const unicode_view& form, const unicode_view& s);
Unicode kernel_unicodedata_category(const unicode_view& s);
#endif

// ord and chr
int64_t kernel_builtins_ord(const string_view& c);
int64_t kernel_builtins_ord(const unicode_view& c);
int64_t kernel_builtins_ord(const Any& c);
Unicode kernel_builtins_chr(int64_t i);

// json
RTValue kernel_json_load(PyArgs args);
RTValue kernel_json_loads(PyArgs args);
// RTValue kernel_json_dump(PyArgs args);
Unicode kernel_json_dumps(PyArgs args);

// file
File kernel_file_open(PyArgs args);
void kernel_file_close(const File& f);

// ndarray global method
NDArray kernel_nd_module_add(const Any& lhs, const Any& rhs);
NDArray kernel_nd_module_sub(const Any& lhs, const Any& rhs);
NDArray kernel_nd_module_div(const Any& lhs, const Any& rhs);
NDArray kernel_nd_module_mul(const Any& lhs, const Any& rhs);
NDArray kernel_nd_module_rand(const Any& view);
NDArray kernel_nd_module_concatenate(PyArgs args);
NDArray kernel_nd_module_stack(PyArgs args);

void kernel_list_module_sort(PyArgs args);
void kernel_list_module_nth_element(PyArgs args);
void kernel_list_module_heapify(PyArgs args);
void kernel_list_module_heap_replace(PyArgs args);
RTValue kernel_list_module_heap_pushpop(PyArgs args);

OpaqueObject kernel_cuda_module_default_stream(int64_t device_id);
OpaqueObject kernel_cuda_module_create_stream(int64_t device_id);
void kernel_cuda_module_stream_sync(const OpaqueObject& stream, int64_t device_id);
void kernel_cuda_module_stream_sync(const Any& stream, int64_t device_id);

// time
double kernel_time_time();

// os
RTValue kernel_os_getenv(PyArgs);

// pickle
Unicode kernel_pickle_serialize(const Any& o);
template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
RTValue kernel_pickle_serialize(const T& o) {
  return kernel_pickle_serialize(RTView(o));
}
RTValue kernel_pickle_deserialize(unicode_view s);
template <typename T, typename = typename std::enable_if<is_runtime_value<T>::value>::type>
RTValue kernel_pickle_deserialize(const T& s) {
  return kernel_pickle_deserialize(s.template As<unicode_view>());
}

// base64
String kernel_base64_b64encode(string_view s, RTView altchars);
String kernel_base64_b64decode(string_view s, RTView altchars, bool validate);

// sorted
List kernel_builtins_sorted(const List& iterable, const Any& key_func, bool reverse);
List kernel_builtins_sorted(const Tuple& iterable, const Any& key_func, bool reverse);
RTValue kernel_builtins_sorted(const Any& iterable, const Any& key_func, bool reverse);

/******************************************************************************
 * python builtin modules and functions
 *
 * Function schema:
 *     RTValue module_method(self, *args);
 *
 *****************************************************************************/

RTValue kernel_math_iterable_min(const List& arg);
RTValue kernel_math_iterable_min(const Set& arg);
RTValue kernel_math_iterable_min(const Any& arg);
RTValue kernel_math_min(PyArgs args);

MATXSCRIPT_ALWAYS_INLINE int64_t kernel_math_int_min(std::initializer_list<int64_t> args) {
  return std::min(args);
}

MATXSCRIPT_ALWAYS_INLINE double kernel_math_double_min(std::initializer_list<double> args) {
  return std::min(args);
}

MATXSCRIPT_ALWAYS_INLINE int64_t kernel_math_int_max(std::initializer_list<int64_t> args) {
  return std::max(args);
}

MATXSCRIPT_ALWAYS_INLINE double kernel_math_double_max(std::initializer_list<double> args) {
  return std::max(args);
}

RTValue kernel_math_iterable_max(const List& arg);
RTValue kernel_math_iterable_max(const Set& arg);
RTValue kernel_math_iterable_max(const Any& arg);
RTValue kernel_math_max(PyArgs args);

RTValue kernel_builtins_print(PyArgs args = {},
                              string_view sep = " ",
                              string_view end = "\n",
                              FILE* file = stdout);

MATXSCRIPT_ALWAYS_INLINE void kernel_builtins_print_one(std::ostream& os, const Any& arg) {
  if (arg.IsString()) {
    auto view = arg.AsNoCheck<string_view>();
    os << "b'" << BytesEscape(view.data(), view.size()) << "'";
  } else if (arg.IsUnicode()) {
    auto view = arg.AsNoCheck<unicode_view>();
    os << view;
  } else {
    os << arg;
  }
}

template <typename ItemType>
MATXSCRIPT_ALWAYS_INLINE void kernel_builtins_print_one(std::ostream& os,
                                                        const FTList<ItemType>& arg) {
  os << arg;
}

template <typename ItemType>
MATXSCRIPT_ALWAYS_INLINE void kernel_builtins_print_one(std::ostream& os,
                                                        const FTSet<ItemType>& arg) {
  os << arg;
}

template <typename KeyType, typename ValueType>
MATXSCRIPT_ALWAYS_INLINE void kernel_builtins_print_one(std::ostream& os,
                                                        const FTDict<KeyType, ValueType>& arg) {
  os << arg;
}

template <typename UnpackArg,
          typename = typename std::enable_if<!is_runtime_value<UnpackArg>::value>::type>
MATXSCRIPT_ALWAYS_INLINE void kernel_builtins_print_one(std::ostream& os, const UnpackArg& arg) {
  GenericValueConverter<RTView> Converter;
  kernel_builtins_print_one(os, static_cast<const Any&>(Converter(arg)));
}

template <size_t I>
struct kernel_builtins_print_details {
  template <typename UnpackArg, class... ARGS>
  static MATXSCRIPT_ALWAYS_INLINE void run(string_view sep,
                                           std::ostream& os,
                                           const UnpackArg& arg,
                                           ARGS&&... args) {
    kernel_builtins_print_one(os, arg);
    os << sep;
    return kernel_builtins_print_details<I - 1>::run(sep, os, std::forward<ARGS>(args)...);
  }
};

template <>
struct kernel_builtins_print_details<1> {
  template <typename UnpackArg, class... ARGS>
  static MATXSCRIPT_ALWAYS_INLINE void run(string_view sep,
                                           std::ostream& os,
                                           const UnpackArg& arg,
                                           ARGS&&... args) {
    kernel_builtins_print_one(os, arg);
  }
};

template <class... ARGS>
MATXSCRIPT_ALWAYS_INLINE RTValue
kernel_builtins_print(const string_view& sep, const string_view& end, FILE* file, ARGS&&... args) {
  std::stringstream os;
  if (sizeof...(args) > 0) {
    kernel_builtins_print_details<sizeof...(args)>::run(sep, os, std::forward<ARGS>(args)...);
  }
  os << end;
  auto repr = os.str();
  fprintf(file, "%*s", (int)repr.size(), repr.c_str());
  return None;
}

template <class TYPE>
MATXSCRIPT_ALWAYS_INLINE bool kernel_builtins_isinstance(const Any& value) {
  return value.template Is<TYPE>();
}

/**
 * UserData methods
 **/
RTValue kernel_object_call(const Any& self, PyArgs args);

// defined in tx_session.cc
RTView GetClosureVar(void* session_handle, const string_view& cls, const string_view& name);

// defined in tx_session.cc
List ParallelMap(const UserDataRef& func, const List& inputs, void* session_handle);
Tuple ParallelMap(const UserDataRef& func, const Tuple& inputs, void* session_handle);
RTValue ParallelMap(const UserDataRef& func, const Any& inputs, void* session_handle);
List ParallelStarMap(const UserDataRef& func, const List& inputs, void* session_handle);
Tuple ParallelStarMap(const UserDataRef& func, const Tuple& inputs, void* session_handle);
RTValue ParallelStarMap(const UserDataRef& func, const Any& inputs, void* session_handle);
RTValue ApplyAsync(const UserDataRef& func, const PyArgs& inputs, void* session_handle);

}  // namespace runtime
}  // namespace matxscript
