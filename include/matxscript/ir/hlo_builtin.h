// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by Halide/TVM IR.
 *
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
/*!
 * \file matx/ir/hlo_builtin.h
 * \brief high level ir builtin intrinsics.
 *
 */
#pragma once

#include <matxscript/ir/op_expr.h>

namespace matxscript {
namespace ir {

/*! \brief Collection of builtin intrinsics as ops */
namespace builtin {
/******************************************************************************
 * make_kwargs_op
 *****************************************************************************/

MATX_DLL const Op& make_kwargs_op();

/******************************************************************************
 * torch ops
 *****************************************************************************/

MATX_DLL const Op& torch_ops();
/******************************************************************************
 * numpy ops
 *****************************************************************************/
MATX_DLL const Op& numpy_ops();

/******************************************************************************
 * lambda function
 *****************************************************************************/
MATX_DLL const Op& call_lambda();

/******************************************************************************
 * builtins functions
 *****************************************************************************/
MATX_DLL const Op& hlo_if_then_else();

/******************************************************************************
 * List builtin functions
 *****************************************************************************/
MATX_DLL const Op& list___len__();
MATX_DLL const Op& list___contains__();
MATX_DLL const Op& list___getitem__();
MATX_DLL const Op& list___setitem__();
MATX_DLL const Op& list___getslice__();
MATX_DLL const Op& list_append();
MATX_DLL const Op& list_extend();
MATX_DLL const Op& list_repeat();
MATX_DLL const Op& list_fused_repeat_one();
MATX_DLL const Op& list_fused_repeat_many();
MATX_DLL const Op& list_reserve();
MATX_DLL const Op& list_index();
MATX_DLL const Op& list_capacity();
MATX_DLL const Op& list_pop();
MATX_DLL const Op& list_insert();
MATX_DLL const Op& list_remove();
MATX_DLL const Op& list_clear();
MATX_DLL const Op& list_reverse();
MATX_DLL const Op& list_count();
MATX_DLL const Op& list_sort_no_key();
MATX_DLL const Op& list_sort();

MATX_DLL const Op& ft_list___len__();
MATX_DLL const Op& ft_list___contains__();
MATX_DLL const Op& ft_list___getitem__();
MATX_DLL const Op& ft_list___setitem__();
MATX_DLL const Op& ft_list___getslice__();
MATX_DLL const Op& ft_list_append();
MATX_DLL const Op& ft_list_extend();
MATX_DLL const Op& ft_list_repeat();
MATX_DLL const Op& ft_list_fused_repeat_one();
MATX_DLL const Op& ft_list_fused_repeat_many();
MATX_DLL const Op& ft_list_reserve();
MATX_DLL const Op& ft_list_index();
MATX_DLL const Op& ft_list_capacity();
MATX_DLL const Op& ft_list_pop();
MATX_DLL const Op& ft_list_insert();
MATX_DLL const Op& ft_list_remove();
MATX_DLL const Op& ft_list_clear();
MATX_DLL const Op& ft_list_reverse();
MATX_DLL const Op& ft_list_count();
MATX_DLL const Op& ft_list_sort_no_key();
MATX_DLL const Op& ft_list_sort();
/******************************************************************************
 * Dict builtin functions
 *****************************************************************************/
MATX_DLL const Op& dict___len__();
MATX_DLL const Op& dict___contains__();
MATX_DLL const Op& dict___getitem__();
MATX_DLL const Op& dict___setitem__();
MATX_DLL const Op& dict_clear();
MATX_DLL const Op& dict_reserve();
MATX_DLL const Op& dict_bucket_count();
MATX_DLL const Op& dict_keys();
MATX_DLL const Op& dict_values();
MATX_DLL const Op& dict_items();
MATX_DLL const Op& dict_get();
MATX_DLL const Op& dict_pop();
/******************************************************************************
 * ADT builtin functions
 *****************************************************************************/
MATX_DLL const Op& tuple_len();
/******************************************************************************
 * Set builtin functions
 *****************************************************************************/
MATX_DLL const Op& set___len__();
MATX_DLL const Op& set___contains__();
MATX_DLL const Op& set_add();
MATX_DLL const Op& set_clear();
MATX_DLL const Op& set_reserve();
MATX_DLL const Op& set_bucket_count();
MATX_DLL const Op& set_difference();
MATX_DLL const Op& set_difference_update();
MATX_DLL const Op& set_update();
MATX_DLL const Op& set_union();
MATX_DLL const Op& set_discard();
/******************************************************************************
 * String builtin functions
 *****************************************************************************/
MATX_DLL const Op& str_lower();
MATX_DLL const Op& str_upper();
MATX_DLL const Op& str_append();
MATX_DLL const Op& str_decode();
/******************************************************************************
 * Unicode builtin functions
 *****************************************************************************/
MATX_DLL const Op& unicode_find();
MATX_DLL const Op& unicode_encode();

/******************************************************************************
 * NDArray builtin functions
 *****************************************************************************/
MATX_DLL const Op& ndarray___getitem__();
MATX_DLL const Op& ndarray_getitem_as_int64();
MATX_DLL const Op& ndarray_getitem_as_double();
MATX_DLL const Op& ndarray___setitem__();
MATX_DLL const Op& ndarray_fused_getitem();
MATX_DLL const Op& ndarray_fused_getitem_as_int64();
MATX_DLL const Op& ndarray_fused_getitem_as_double();
MATX_DLL const Op& ndarray_fused_setitem();

/******************************************************************************
 * Fused functions
 *****************************************************************************/
MATX_DLL const Op& str_fused_concat();
MATX_DLL const Op& unicode_fused_concat();

/******************************************************************************
 * UserData dispatch
 *****************************************************************************/
MATX_DLL const Op& object___getitem__();
MATX_DLL const Op& object___setitem__();
MATX_DLL const Op& object___fused_getitem__();
MATX_DLL const Op& object___fused_setitem__();
MATX_DLL const Op& object___dispatch__();
MATX_DLL const Op& object___getattr__();
MATX_DLL const Op& object___setattr__();
MATX_DLL const Op& user_data_get_attr();
MATX_DLL const Op& user_data_set_attr();
MATX_DLL const Op& user_data_call();
MATX_DLL const Op& user_data_call_attr();
/******************************************************************************
 * Generic Container builtin functions
 *****************************************************************************/
MATX_DLL const Op& object_append();
MATX_DLL const Op& object_slice_append();
MATX_DLL const Op& object_contains();
MATX_DLL const Op& object_slice_contains();
MATX_DLL const Op& object_add();
MATX_DLL const Op& object_extend();
MATX_DLL const Op& object_slice_add();
MATX_DLL const Op& object_clear();
MATX_DLL const Op& object_slice_clear();
MATX_DLL const Op& object_get_item();
MATX_DLL const Op& object_slice_get_item();
MATX_DLL const Op& object_set_item();
MATX_DLL const Op& object_slice_set_item();

MATX_DLL const Op& object_slice_load();
MATX_DLL const Op& object_slice_store();

MATX_DLL const Op& object_find();

MATX_DLL const Op& object_slice_lower();
MATX_DLL const Op& object_slice_upper();
MATX_DLL const Op& object_slice_isdigit();
MATX_DLL const Op& object_slice_isalpha();

MATX_DLL const Op& builtins_print();
MATX_DLL const Op& object_call();

MATX_DLL const Op& builtins_unpack();

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
