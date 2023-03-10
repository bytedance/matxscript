// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the expressions is inspired by TVM.
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
#include <matxscript/ir/hlo_builtin.h>

#include <matxscript/ir/hlo_expr.h>
#include "./hlo_builtin_macros.h"

namespace matxscript {
namespace ir {
namespace builtin {

/******************************************************************************
 * NDArray builtin methods
 *****************************************************************************/
MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_to_list, ToList)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_tolist, ToList)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_is_contiguous, IsContiguous)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_contiguous, Contiguous)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_reshape, Reshape)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("shape", "List|Tuple|Any|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_squeeze, Squeeze)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("axis", "Tuple|Any|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_unsqueeze, Unsqueeze)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("dim", "int|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_shape, ShapeList)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_dtype, DTypeUnicode)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_dim, GetDim)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_device, Device)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray___getitem__, get_item)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "int|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_getitem_as_double, get_item_as_double)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "int|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_getitem_as_int64, get_item_as_int64)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "int|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_fused_getitem, fused_get_item)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_fused_getitem_as_double,
                                               fused_get_item_as_double)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_fused_getitem_as_int64,
                                               fused_get_item_as_int64)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "<template>", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray___setitem__, set_item)
    .set_num_inputs(3)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "int|any_view", "")
    .add_argument("item", "int|float|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_fused_setitem, fused_set_item)
    .set_num_inputs(3)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("index", "<template>", "")
    .add_argument("item", "int|float|any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray___getslice__, get_slice)
    .set_num_inputs(4)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("begin", "int", "")
    .add_argument("end", "int", "")
    .add_argument("step", "int", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray___setslice__, set_slice)
    .set_num_inputs(4)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("begin", "int", "")
    .add_argument("end", "int", "")
    .add_argument("item", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray___len__, size)
    .set_num_inputs(1)
    .add_argument("self", "matx.NDArray", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_transpose, transpose)
    .set_num_inputs(1)
    .set_num_inputs_max(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("axes", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_EXPLICIT(ndarray_as_type, as_type)
    .set_num_inputs(2)
    .add_argument("self", "matx.NDArray", "")
    .add_argument("dtype_str", "unicode_view", "");

/******************************************************************************
 * NDArray global functions
 *****************************************************************************/

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(nd, module_add)
    .set_num_inputs(2)
    .add_argument("lhs", "any_view", "")
    .add_argument("rhs", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(nd, module_sub)
    .set_num_inputs(2)
    .add_argument("lhs", "any_view", "")
    .add_argument("rhs", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(nd, module_div)
    .set_num_inputs(2)
    .add_argument("lhs", "any_view", "")
    .add_argument("rhs", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(nd, module_mul)
    .set_num_inputs(2)
    .add_argument("lhs", "any_view", "")
    .add_argument("rhs", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(nd, module_rand)
    .set_num_inputs(1)
    .add_argument("lhs", "any_view", "")
    .add_argument("rhs", "any_view", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(nd, module_concatenate)
    .set_num_inputs(1)
    .add_argument("args", "*args", "");

MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC_GENERIC(nd, module_stack)
    .set_num_inputs(1)
    .add_argument("args", "*args", "");

static runtime::RTValue TryFusedNDArrayGetItem(BaseExpr container, BaseExpr index) {
  if (auto* call_node = container.as<CallNode>()) {
    if (call_node->op.same_as(ndarray___getitem__())) {
      if (IsIntegerType(index->checked_type_) || IsObjectType(index->checked_type_)) {
        index = HLOCastPrim(runtime::DataType(kDLInt, 64, 1), index);
      } else {
        MXTHROW << "[__getitem__] key must be int type";
      }
      Array<BaseExpr> new_args = {HLOCastPrim(runtime::DataType(kDLInt, 64, 1), call_node->args[1]),
                                  std::move(index)};
      InitializerList call_args(std::move(new_args));
      return Call(ObjectType(),
                  builtin::ndarray_fused_getitem(),
                  {call_node->args[0], std::move(call_args)},
                  container->span,
                  {});
    } else if (call_node->op.same_as(ndarray_fused_getitem())) {
      if (IsIntegerType(index->checked_type_) || IsObjectType(index->checked_type_)) {
        index = HLOCastPrim(runtime::DataType(kDLInt, 64, 1), index);
      } else {
        MXTHROW << "[__getitem__] key must be int type";
      }
      InitializerList old_call_args = runtime::Downcast<InitializerList>(call_node->args[1]);
      Array<BaseExpr> new_args;
      for (auto& oa : old_call_args->fields) {
        new_args.push_back(oa);
      }
      new_args.push_back(std::move(index));
      InitializerList call_args(std::move(new_args));
      return Call(ObjectType(),
                  builtin::ndarray_fused_getitem(),
                  {call_node->args[0], std::move(call_args)},
                  container->span,
                  {});
    }
  }
  return runtime::None;
}

static runtime::RTValue TryFusedNDArraySetItem(BaseExpr container, BaseExpr index, BaseExpr value) {
  if (auto* call_node = container.as<CallNode>()) {
    if (call_node->op.same_as(ndarray___getitem__())) {
      if (IsIntegerType(index->checked_type_) || IsObjectType(index->checked_type_)) {
        index = HLOCastPrim(runtime::DataType(kDLInt, 64, 1), index);
      } else {
        MXTHROW << "[__getitem__] key must be int type";
      }
      Array<BaseExpr> new_args = {HLOCastPrim(runtime::DataType(kDLInt, 64, 1), call_node->args[1]),
                                  std::move(index)};
      InitializerList call_args(std::move(new_args));
      return Call(ObjectType(),
                  builtin::ndarray_fused_setitem(),
                  {call_node->args[0], std::move(call_args), std::move(value)},
                  container->span,
                  {});
    } else if (call_node->op.same_as(ndarray_fused_getitem())) {
      if (IsIntegerType(index->checked_type_) || IsObjectType(index->checked_type_)) {
        index = HLOCastPrim(runtime::DataType(kDLInt, 64, 1), index);
      } else {
        MXTHROW << "[__getitem__] key must be int type";
      }
      InitializerList old_call_args = runtime::Downcast<InitializerList>(call_node->args[1]);
      Array<BaseExpr> new_args;
      for (auto& oa : old_call_args->fields) {
        new_args.push_back(oa);
      }
      new_args.push_back(std::move(index));
      InitializerList call_args(std::move(new_args));
      return Call(ObjectType(),
                  builtin::ndarray_fused_setitem(),
                  {call_node->args[0], std::move(call_args), std::move(value)},
                  container->span,
                  {});
    }
  }
  return runtime::None;
}

static runtime::RTValue TryNDArrayItemAsDouble(BaseExpr item) {
  if (auto* call_node = item.as<CallNode>()) {
    if (call_node->op.same_as(ndarray___getitem__())) {
      return Call(PrimType(runtime::DataType::Float(64)),
                  builtin::ndarray_getitem_as_double(),
                  call_node->args,
                  call_node->span,
                  call_node->type_args);
    } else if (call_node->op.same_as(ndarray_fused_getitem())) {
      return Call(PrimType(runtime::DataType::Float(64)),
                  builtin::ndarray_fused_getitem_as_double(),
                  call_node->args,
                  call_node->span,
                  call_node->type_args);
    }
  }
  return runtime::None;
}

static runtime::RTValue TryNDArrayItemAsInt64(BaseExpr item) {
  if (auto* call_node = item.as<CallNode>()) {
    if (call_node->op.same_as(ndarray___getitem__())) {
      return Call(PrimType(runtime::DataType::Int(64)),
                  builtin::ndarray_getitem_as_int64(),
                  call_node->args,
                  call_node->span,
                  call_node->type_args);
    } else if (call_node->op.same_as(ndarray_fused_getitem())) {
      return Call(PrimType(runtime::DataType::Int(64)),
                  builtin::ndarray_fused_getitem_as_int64(),
                  call_node->args,
                  call_node->span,
                  call_node->type_args);
    }
  }
  return runtime::None;
}

MATXSCRIPT_REGISTER_GLOBAL("ir.TryFusedNDArrayGetItem").set_body_typed(TryFusedNDArrayGetItem);
MATXSCRIPT_REGISTER_GLOBAL("ir.TryFusedNDArraySetItem").set_body_typed(TryFusedNDArraySetItem);
MATXSCRIPT_REGISTER_GLOBAL("ir.TryNDArrayItemAsDouble").set_body_typed(TryNDArrayItemAsDouble);
MATXSCRIPT_REGISTER_GLOBAL("ir.TryNDArrayItemAsInt64").set_body_typed(TryNDArrayItemAsInt64);

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
