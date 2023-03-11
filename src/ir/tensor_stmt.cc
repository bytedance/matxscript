// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by TVM TensorIR.
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
 * \file matx/ir/tensor_stmt.cc
 */
#include <matxscript/ir/tensor_stmt.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/prim_ops.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/type_helper_macros.h>

#include <utility>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::PyArgs;
using ::matxscript::runtime::RTValue;
using ::matxscript::runtime::RTView;

// Buffer
Buffer::Buffer(PrimVar data,
               runtime::DataType dtype,
               Array<PrimExpr> shape,
               Array<PrimExpr> strides,
               PrimExpr elem_offset,
               StringRef name,
               int data_alignment,
               int offset_factor,
               BufferType buffer_type,
               Array<IntImm> axis_separators,
               Span span) {
  runtime::DataType storage_dtype = dtype;
  // specially handle bool
  if (storage_dtype == runtime::DataType::Bool()) {
    storage_dtype = runtime::DataType::Int(8);
  }
  // The buffer dtype may differ from the dtype of the underlying
  // allocation, such as a single allocation that backs multiple
  // tensors without a common datatype.  Therefore, we check that the
  // data pointer is a pointer, but not the exact type of the
  // pointed-to values.

  // TODO(Lunderberg): Use an explicit pointer cast for the data
  // pointer.  Should be done alongside extensions to StmtExprMutator
  // to more easily handle buffer/buffer_var updates.
  MXCHECK(data->type_annotation.defined())
      << "Variable " << data->name_hint << " is missing a type annotation.";
  MXCHECK(data->type_annotation.as<PointerTypeNode>())
      << "Variable " << data->name_hint << " is not a pointer.";
  MXCHECK(data->type_annotation.as<PointerTypeNode>()->element_type.as<PrimTypeNode>())
      << "Variable " << data->name_hint << " does not point to a primitive.";

  auto n = runtime::make_object<BufferNode>();
  n->data = std::move(data);
  n->dtype = dtype;

  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->axis_separators = std::move(axis_separators);
  n->name = std::move(name);
  if (!elem_offset.defined()) {
    elem_offset = make_const(n->DefaultIndexType(), 0);
  }
  if (data_alignment <= 0) {
    data_alignment = runtime::kAllocAlignment;
  }
  if (offset_factor == 0) {
    offset_factor = 1;
  }
  n->elem_offset = std::move(elem_offset);
  n->data_alignment = data_alignment;
  n->offset_factor = offset_factor;
  n->buffer_type = buffer_type;
  if (n->buffer_type == kAutoBroadcast && n->shape.size() > 0 && n->strides.empty()) {
    for (size_t i = 0; i < n->shape.size(); ++i) {
      n->strides.push_back(PrimVar("stride", n->shape[i].dtype()));
    }
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(BufferNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.Buffer").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 11);
  auto buffer_type = args[8].As<StringRef>();
  BufferType type = (buffer_type == "auto_broadcast") ? kAutoBroadcast : kDefault;
  return Buffer(MATXSCRIPT_TYPE_AS(args[0], PrimVar),
                args[1].As<runtime::DataType>(),
                MATXSCRIPT_TYPE_AS(args[2], Array<PrimExpr>),
                MATXSCRIPT_TYPE_AS(args[3], Array<PrimExpr>),
                MATXSCRIPT_TYPE_AS(args[4], PrimExpr),
                args[5].AsObjectRef<StringRef>(),
                MATXSCRIPT_TYPE_AS(args[6], int),
                MATXSCRIPT_TYPE_AS(args[7], int),
                type,
                MATXSCRIPT_TYPE_AS(args[9], Array<IntImm>),
                MATXSCRIPT_TYPE_AS(args[10], Span));
});

// BufferRegion
BufferRegion::BufferRegion(Buffer buffer, Array<RangeExpr> region) {
  MXCHECK_EQ(buffer->shape.size(), region.size())
      << "The dimension between " << buffer << " and region " << region
      << " mismatched, the buffer is " << buffer;
  ObjectPtr<BufferRegionNode> node = runtime::make_object<BufferRegionNode>();
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

BufferRegion BufferRegion::FullRegion(Buffer buffer) {
  Array<RangeExpr> region;
  for (PrimExpr extent : buffer->shape) {
    region.push_back(RangeExpr(0, extent, 1));
  }
  return BufferRegion(std::move(buffer), std::move(region));
}

BufferRegion BufferRegion::FromPoint(Buffer buffer, Array<PrimExpr> indices) {
  Array<RangeExpr> region;
  for (const PrimExpr& index : indices) {
    region.push_back(RangeExpr(index, add(index, 1), 1));
  }
  return BufferRegion(std::move(buffer), std::move(region));
}

MATXSCRIPT_REGISTER_GLOBAL("ir.BufferRegion")
    .set_body_typed([](Buffer buffer, Array<RangeExpr> region) {
      return BufferRegion(std::move(buffer), std::move(region));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(BufferRegionNode);

// MatchBufferRegion
MatchBufferRegion::MatchBufferRegion(Buffer buffer, BufferRegion source) {
  MXTHROW << "NotImplementedError: MatchBufferRegion";
}

MATXSCRIPT_REGISTER_GLOBAL("ir.MatchBufferRegion")
    .set_body_typed([](Buffer buffer, BufferRegion source) {
      return MatchBufferRegion(std::move(buffer), std::move(source));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(MatchBufferRegionNode);

// ComputeBlock
ComputeBlock::ComputeBlock(Array<PrimIterVar> iter_vars,
                           Array<BufferRegion> reads,
                           Array<BufferRegion> writes,
                           StringRef name_hint,
                           Stmt body,
                           Optional<Stmt> init,
                           Array<Buffer> alloc_buffers,
                           Array<MatchBufferRegion> match_buffers,
                           Map<StringRef, ObjectRef> annotations,
                           Span span) {
  ObjectPtr<ComputeBlockNode> node = runtime::make_object<ComputeBlockNode>();
  node->iter_vars = std::move(iter_vars);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->name_hint = std::move(name_hint);
  node->body = std::move(body);
  node->init = std::move(init);
  node->alloc_buffers = std::move(alloc_buffers);
  node->match_buffers = std::move(match_buffers);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ComputeBlockNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ComputeBlock")
    .set_body_typed([](Array<PrimIterVar> iter_vars,
                       Array<BufferRegion> reads,
                       Array<BufferRegion> writes,
                       StringRef name_hint,
                       Stmt body,
                       Optional<Stmt> init,
                       Array<Buffer> alloc_buffers,
                       Array<MatchBufferRegion> match_buffers,
                       Map<StringRef, ObjectRef> annotations,
                       Span span) {
      return ComputeBlock(std::move(iter_vars),
                          std::move(reads),
                          std::move(writes),
                          std::move(name_hint),
                          std::move(body),
                          std::move(init),
                          std::move(alloc_buffers),
                          std::move(match_buffers),
                          std::move(annotations),
                          std::move(span));
    });

// ComputeBlockRealize
ComputeBlockRealize::ComputeBlockRealize(Array<PrimExpr> values,
                                         PrimExpr predicate,
                                         ComputeBlock block,
                                         Span span) {
  MXCHECK_EQ(block->iter_vars.size(), values.size())
      << "ValueError: ComputeBlockRealize needs to have the same number of iter_vars and binding values";
  MXCHECK(predicate.dtype().is_bool())
      << "TypeError: Expect Block.predicate to be a bool expression";
  ObjectPtr<ComputeBlockRealizeNode> node = runtime::make_object<ComputeBlockRealizeNode>();
  node->iter_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.ComputeBlockRealize")
    .set_body_typed(
        [](Array<PrimExpr> iter_values, PrimExpr predicate, ComputeBlock block, Span span) {
          return ComputeBlockRealize(
              std::move(iter_values), std::move(predicate), std::move(block), std::move(span));
        });

MATXSCRIPT_REGISTER_NODE_TYPE(ComputeBlockRealizeNode);

}  // namespace ir
}  // namespace matxscript
