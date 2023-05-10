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

#include <matxscript/ir/_base/cow_array_ref.h>
#include <matxscript/ir/_base/cow_map_ref.h>
#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/ir/_base/with.h>
#include <matxscript/ir/prim_ops.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/ir/printer/ir_frame.h>
#include <matxscript/ir/printer/utils.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/type_helper_macros.h>

#include <utility>

namespace matxscript {
namespace ir {

using ::matxscript::runtime::Downcast;
using ::matxscript::runtime::GetRef;
using ::matxscript::runtime::PyArgs;
using ::matxscript::runtime::RTValue;
using ::matxscript::runtime::RTView;

using namespace ::matxscript::ir::printer;

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
  //MXCHECK(data->type_annotation.as<PointerTypeNode>()->element_type.as<PrimTypeNode>())
  //    << "Variable " << data->name_hint << " does not point to a primitive.";

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

PrimExpr Buffer::vload(Array<PrimExpr> begin, DataType value_dtype) const {
  // specially handle bool, stored as DataType::Int(8)
  const BufferNode* n = operator->();
  MXCHECK(n != nullptr);
  MXCHECK(value_dtype.element_of() == n->dtype.element_of() &&
          value_dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot load " << value_dtype << " from buffer of " << n->dtype;

  Array<PrimExpr> indices = begin;
  int factor = value_dtype.lanes() / n->dtype.lanes();
  if (factor > 1) {
    MXTHROW << "ramp is not supported!!!";
    // indices.Set(indices.size() - 1, Ramp(indices[indices.size() - 1], 1, factor));
  }
  return BufferLoad(*this, indices);
}

Stmt Buffer::vstore(Array<PrimExpr> begin, PrimExpr value) const {
  // specially handle bool, stored as DataType::Int(8)
  const BufferNode* n = operator->();
  MXCHECK(n != nullptr);
  DataType value_dtype = value.dtype();
  MXCHECK(value_dtype.element_of() == n->dtype.element_of() &&
          value_dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot store " << value_dtype << " to buffer of " << n->dtype;

  Array<PrimExpr> indices = begin;
  int factor = value_dtype.lanes() / n->dtype.lanes();
  if (factor > 1) {
    MXTHROW << "ramp is not supported!!!";
    // indices.Set(indices.size() - 1, Ramp(indices[indices.size() - 1], 1, factor));
  }
  return BufferStore(*this, value, indices);
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
                args[5].As<runtime::String>(),
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

// BufferLoad
void BufferLoadNode::LegalizeDType() {
  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    MXCHECK(indices[i].dtype().is_scalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  int index_lanes = indices.size() ? indices.back().dtype().lanes() : 1;
  int buffer_lanes = buffer->dtype.lanes();

  this->dtype = buffer->dtype.with_lanes(index_lanes * buffer_lanes);
}

BufferLoad::BufferLoad(Buffer buffer, Array<PrimExpr> indices, Span span) {
  MXCHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  ObjectPtr<BufferLoadNode> node = runtime::make_object<BufferLoadNode>();
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  node->span = std::move(span);
  node->LegalizeDType();
  node->checked_type_ = PrimType(node->dtype);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.BufferLoad")
    .set_body_typed([](Buffer buffer, Array<PrimExpr> indices, Span span) {
      return BufferLoad(buffer, indices, span);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(BufferLoadNode);

// BufferStore
BufferStore::BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
  MXCHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    MXCHECK(indices[i].dtype().is_scalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  int index_lanes = indices.size() ? indices.back().dtype().lanes() : 1;
  int buffer_lanes = buffer->dtype.lanes();

  MXCHECK_EQ(index_lanes * buffer_lanes, value.dtype().lanes())
      << "Cannot store value with " << value.dtype().lanes() << ", expected value with "
      << index_lanes * buffer_lanes << " (" << index_lanes << " index lanes * " << buffer_lanes
      << " buffer element lanes)";
  if (buffer->dtype.with_lanes(buffer_lanes * index_lanes) != value.dtype()) {
    MXLOG(FATAL) << "TypeError: dtype mismatch on BufferStore: "      //
                 << "buffer's dtype is `" << buffer->dtype            //
                 << "`, the lanes of indexing are: `" << index_lanes  //
                 << "`, but RHS's dtype is `" << value.dtype() << "`";
  }

  ObjectPtr<BufferStoreNode> node = runtime::make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.BufferStore")
    .set_body_typed([](Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
      return BufferStore(buffer, value, indices, span);
    });

MATXSCRIPT_REGISTER_NODE_TYPE(BufferStoreNode);

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

Map<StringRef, ExprDoc> BufferAttrs(ir::Buffer buffer,
                                    const ObjectPath& buffer_p,
                                    const Frame& frame,
                                    const IRDocsifier& d) {
  Map<StringRef, ExprDoc> kwargs;
  Array<ExprDoc> var_def_lhs;
  Array<ExprDoc> var_def_rhs;

  // Step 0. Set up statistics
  std::unordered_map<const Object*, int> use_count;
  auto update_use_count = [&](const PrimExpr& e) {
    ir::PostOrderVisit(e, [&](const ObjectRef& n) {
      if (const PrimVarNode* var = n.as<PrimVarNode>()) {
        ++use_count[var];
      }
    });
  };
  update_use_count(buffer->elem_offset);
  update_use_count(buffer->data);
  for (const PrimExpr& e : buffer->strides) {
    update_use_count(e);
  }
  for (const PrimExpr& e : buffer->shape) {
    update_use_count(e);
  }
  auto is_new_var = [&](const PrimExpr& e) {
    return e->IsInstance<PrimVarNode>() && !d->IsVarDefined(e);
  };
  auto add_out_of_line_var_def = [&](const PrimVar& var, const ObjectPath& var_p) {
    MXCHECK(!d->IsVarDefined(var));
    ExprDoc lhs = DefineVar(var, frame, d);
    lhs->source_paths.push_back(var_p);
    var_def_lhs.push_back(lhs);
    var_def_rhs.push_back(PrintVarCreation(var, var_p, d));
  };
  auto try_inline_def =
      [&](const PrimExpr& e, const ObjectPath& e_p, std::function<ExprDoc()> inline_f) {
        MXCHECK(is_new_var(e));
        PrimVar var = Downcast<PrimVar>(e);
        if (use_count[var.get()] == 1) {
          d->Define(e, frame, inline_f);
          return true;
        } else {
          add_out_of_line_var_def(var, e_p);
          return false;
        }
      };
  // Step 1. Handle `buffer.shape`
  {
    const Array<PrimExpr>& shape = buffer->shape;
    ObjectPath shape_p = buffer_p->Attr("shape");
    int n = shape.size();
    Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = shape[i];
      ObjectPath e_p = shape_p->ArrayIndex(i);
      if (is_new_var(e)) {
        add_out_of_line_var_def(Downcast<PrimVar>(e), e_p);
      }
      results.push_back(d->AsDoc<ExprDoc>(e, e_p));
    }
    kwargs.Set("shape", TupleDoc(results));
  }
  // Step 2. Handle `buffer.dtype`
  kwargs.Set("dtype", LiteralDoc::DataType(buffer->dtype, buffer_p->Attr("dtype")));
  // Step 3. Handle `buffer.data`
  if (!is_new_var(buffer->data)) {
    kwargs.Set("data", d->AsDoc<ExprDoc>(buffer->data, buffer_p->Attr("data")));
  } else {
    try_inline_def(buffer->data, buffer_p->Attr("data"), [=]() {
      return d->AsDoc<ExprDoc>(buffer, buffer_p)->Attr("data");
    });
  }
  // Step 4. Handle `buffer.strides`
  if (!buffer->strides.empty()) {
    const Array<PrimExpr>& strides = buffer->strides;
    ObjectPath strides_p = buffer_p->Attr("strides");
    int n = strides.size();
    Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = strides[i];
      ObjectPath e_p = strides_p->ArrayIndex(i);
      if (is_new_var(e)) {
        if (try_inline_def(e, e_p, [=]() {
              return d->AsDoc<ExprDoc>(buffer, buffer_p)
                  ->Attr("strides")[{LiteralDoc::Int(i, NullOpt)}];
            })) {
          results.push_back(LiteralDoc::Str(Downcast<PrimVar>(e)->name_hint, e_p));
          continue;
        }
      }
      results.push_back(d->AsDoc<ExprDoc>(e, e_p));
    }
    kwargs.Set("strides", TupleDoc(results));
  }
  // Step 5. Handle `buffer.elem_offset`
  bool needs_print_factor = false;
  if (const auto* int_imm = buffer->elem_offset.as<IntImmNode>()) {
    if (int_imm->value != 0) {
      kwargs.Set("elem_offset",
                 d->AsDoc<ExprDoc>(buffer->elem_offset,  //
                                   buffer_p->Attr("elem_offset")));
    }
  } else if (is_new_var(buffer->elem_offset)) {
    try_inline_def(buffer->elem_offset, buffer_p->Attr("elem_offset"), [=]() {
      return d->AsDoc<ExprDoc>(buffer, buffer_p)->Attr("elem_offset");
    });
    needs_print_factor = true;
  } else {
    kwargs.Set("elem_offset",
               d->AsDoc<ExprDoc>(buffer->elem_offset,  //
                                 buffer_p->Attr("elem_offset")));
  }
  // Step 6. Handle `buffer.scope`
  // Step 7. Handle `buffer.data_alignment`
  if (buffer->data_alignment != runtime::kAllocAlignment) {
    kwargs.Set("align", LiteralDoc::Int(buffer->data_alignment, buffer_p->Attr("data_alignment")));
  }
  // Step 8. Handle `buffer.offset_factor`
  if (needs_print_factor || buffer->offset_factor != 1) {
    kwargs.Set("offset_factor",
               LiteralDoc::Int(buffer->offset_factor, buffer_p->Attr("offset_factor")));
  }
  // Step 9. Handle `buffer.buffer_type`
  if (buffer->buffer_type != ir::BufferType::kDefault) {
    kwargs.Set("buffer_type", LiteralDoc::Str("auto", buffer_p->Attr("buffer_type")));
  }
  // Step 10. Handle `buffer.axis_separator`
  if (!buffer->axis_separators.empty()) {
    kwargs.Set("axis_separators",
               d->AsDoc<ExprDoc>(buffer->axis_separators, buffer_p->Attr("axis_separators")));
  }
  if (var_def_lhs.size() == 1) {
    frame->stmts.push_back(AssignDoc(var_def_lhs[0], var_def_rhs[0], NullOpt));
  } else if (var_def_lhs.size() > 1) {
    frame->stmts.push_back(AssignDoc(TupleDoc(var_def_lhs), TupleDoc(var_def_rhs), NullOpt));
  }
  return kwargs;
}

ExprDoc BufferCall(const ExprDoc& prefix,
                   const Map<StringRef, ExprDoc>& attrs,
                   Array<ExprDoc> args) {
  Array<StringRef> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  for (StringRef s : {"shape", "dtype"}) {
    if (Optional<ExprDoc> doc = attrs.Get(s)) {
      args.push_back(doc.value());
    }
  }
  for (StringRef s : {"data",
                      "strides",
                      "elem_offset",
                      "scope",
                      "align",
                      "offset_factor",
                      "buffer_type",
                      "axis_separators"}) {
    if (Optional<ExprDoc> doc = attrs.Get(s)) {
      kwargs_keys.push_back(s);
      kwargs_values.push_back(doc.value());
    }
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

ExprDoc BufferDecl(const ir::Buffer& buffer,
                   const StringRef& method,
                   const Array<ExprDoc>& args,
                   const ObjectPath& p,
                   const Frame& frame,
                   const IRDocsifier& d) {
  return BufferCall(/*prefix=*/Dialect(d, method),
                    /*attrs=*/BufferAttrs(buffer, p, frame, d),
                    /*args=*/args);
}

ExprDoc BufferAttn(const ir::Buffer& buffer,
                   const ObjectPath& p,
                   const Frame& frame,
                   const IRDocsifier& d) {
  Map<StringRef, ExprDoc> attrs = BufferAttrs(buffer, p, frame, d);
  ExprDoc shape = attrs.Get("shape").value();
  ExprDoc dtype =
      attrs.Get("dtype").value_or(LiteralDoc::DataType(buffer->dtype, p->Attr("dtype")));
  return Dialect(d, "Buffer")->Call({shape, dtype}, {}, {});
}

Array<Doc> BufferIndices(const Array<PrimExpr>& indices,
                         const ObjectPath& p,
                         const IRDocsifier& d) {
  int n = indices.size();
  Array<Doc> indices_doc;
  indices_doc.reserve(n);
  for (int i = 0; i < n; ++i) {
    indices_doc.push_back(d->AsDoc<ExprDoc>(indices[i], p->Attr("indices")->ArrayIndex(i)));
  }
  return indices_doc;
}

Array<Doc> BufferSlices(const Array<RangeExpr>& region, const ObjectPath& p, const IRDocsifier& d) {
  int n = region.size();
  Array<Doc> indices;
  indices.reserve(n);
  for (int i = 0; i < n; ++i) {
    RangeExpr range = region[i];
    ObjectPath range_p = p->ArrayIndex(i);
    ExprDoc start = d->AsDoc<ExprDoc>(range->start, range_p->Attr("start"));
    ExprDoc stop = d->AsDoc<ExprDoc>(range->stop, range_p->Attr("stop"));
    ExprDoc step = d->AsDoc<ExprDoc>(range->step, range_p->Attr("step"));
    indices.push_back(start);
    indices.push_back(stop);
    indices.push_back(step);
  }
  return indices;
}

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::BufferRegion>(
        "", [](ir::BufferRegion buffer_region, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = d->AsDoc<ExprDoc>(buffer_region->buffer, p->Attr("buffer"));
          return prefix[BufferSlices(buffer_region->region, p->Attr("region"), d)];
        });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<ir::Buffer>("", [](ir::Buffer buffer, ObjectPath p, IRDocsifier d) -> Doc {
      if (!d->IsVarDefined(buffer)) {
        if (Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
          ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
          ExprDoc rhs = BufferDecl(buffer, "Buffer", {}, p, opt_f.value(), d);
          opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
      }
      if (Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
        return doc.value();
      }
      MXLOG(FATAL) << "IndexError: Buffer is not defined in the environment: " << buffer;
      return Doc{nullptr};
    });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::MatchBufferRegion>(
        "", [](ir::MatchBufferRegion stmt, ObjectPath p, IRDocsifier d) -> Doc {
          Frame frame = d->frames.back();
          ExprDoc lhs = DefineBuffer(stmt->buffer, frame, d);
          ExprDoc src_buffer = d->AsDoc<ExprDoc>(stmt->source, p->Attr("source"));
          ExprDoc rhs = BufferDecl(
              stmt->buffer, "match_buffer", {src_buffer}, p->Attr("buffer"), d->frames.back(), d);
          return AssignDoc(lhs, rhs, NullOpt);
        });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::BufferLoad>(  //
        "",
        [](ir::BufferLoad load, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(load->buffer, p->Attr("buffer"));
          return buffer[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::BufferStore>(  //
        "",
        [](ir::BufferStore store, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(store->buffer, p->Attr("buffer"));
          return AssignDoc(/*lhs=*/buffer[BufferIndices(store->indices, p->Attr("indices"), d)],
                           /*rhs=*/d->AsDoc<ExprDoc>(store->value, p->Attr("value")),
                           NullOpt);
        });

MATXSCRIPT_REGISTER_GLOBAL("ir.BufferVLoad")
    .set_body_typed([](const Buffer& buf, Array<PrimExpr> begin, runtime::DataType dtype) {
      return buf.vload(begin, dtype);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.BufferVStore")
    .set_body_typed([](const Buffer& buf, Array<PrimExpr> begin, PrimExpr value) {
      return buf.vstore(begin, value);
    });

Doc PrintBlock(IRDocsifier d,
               ir::ComputeBlock block,
               ObjectPath block_p,  //
               Optional<ir::ComputeBlockRealize> opt_realize,
               Optional<ObjectPath> opt_realize_p) {
  With<IRFrame> frame(d, block);
  MXCHECK_EQ(opt_realize.defined(), opt_realize_p.defined());
  const ir::ComputeBlockRealizeNode* realize =
      opt_realize.defined() ? opt_realize.value().get() : nullptr;
  const ObjectPathNode* realize_p = opt_realize_p.defined() ? opt_realize_p.get() : nullptr;
  // Step 1. Handle block var and block bindings
  // Step 1.1. Obtain all loop var defined along path
  std::unordered_map<const ir::PrimVarNode*, ir::For> loop_vars;
  for (Frame f : d->frames) {
    if (const auto* tir_f = f.as<IRFrameNode>()) {
      if (const auto* for_loop = tir_f->tir.as<ir::ForNode>()) {
        for (const ir::ForNode* l = for_loop; l != nullptr; l = l->body.as<ir::ForNode>()) {
          loop_vars.insert(std::make_pair(l->loop_var.get(), GetRef<ir::For>(l)));
        }
      }
    }
  }

  std::vector<int> remap_vars_indices;
  auto add_remapped_iter_var = [&](int i) -> bool { return false; };

  auto print_single_iter_var = [&](int i) {
    ir::PrimIterVar iter_var = block->iter_vars[i];
    ObjectPath iter_var_p = block_p->Attr("iter_var")->ArrayIndex(i);
    ExprDoc rhs = Dialect(d, "axis");
    ExprDoc start = d->AsDoc<ExprDoc>(iter_var->dom->start, iter_var_p->Attr("dom")->Attr("start"));
    ExprDoc stop = d->AsDoc<ExprDoc>(iter_var->dom->stop, iter_var_p->Attr("dom")->Attr("stop"));
    ExprDoc step = d->AsDoc<ExprDoc>(iter_var->dom->step, iter_var_p->Attr("dom")->Attr("step"));
    ExprDoc dom = TupleDoc({start, stop, step});
    if (realize) {
      ExprDoc binding = d->AsDoc<ExprDoc>(realize->iter_values[i],  //
                                          realize_p->Attr("iter_values")->ArrayIndex(i));
      rhs = rhs->Call({dom, binding});
    } else {
      rhs = rhs->Call({dom});
    }
    (*frame)->stmts.push_back(AssignDoc(DefineVar(iter_var->var, *frame, d), rhs, NullOpt));
  };

  auto print_remapped_iter_var = [&]() {
    if (remap_vars_indices.size()) {
      int m = remap_vars_indices.size();
      if (!m) {
        return;
      }
      if (m == 1) {
        print_single_iter_var(remap_vars_indices[0]);
        remap_vars_indices.clear();
        return;
      }
      Array<ExprDoc> lhs;
      Array<ExprDoc> loop_var_doc;
      lhs.reserve(m);
      loop_var_doc.reserve(m);
      for (int i : remap_vars_indices) {
        ir::PrimIterVar iter_var = block->iter_vars[i];
        ObjectPath iter_var_p = block_p->Attr("iter_vars")->ArrayIndex(i);
        lhs.push_back(DefineVar(iter_var->var, *frame, d));
        loop_var_doc.push_back(d->AsDoc<ExprDoc>(realize->iter_values[i],
                                                 realize_p->Attr("iter_values")->ArrayIndex(i)));
      }
      ExprDoc rhs = Dialect(d, "axis")->Attr("remap");
      rhs = rhs->Call({ListDoc(loop_var_doc)});
      (*frame)->stmts.push_back(AssignDoc(TupleDoc(lhs), rhs, NullOpt));
      remap_vars_indices.clear();
    }
  };

  // Step 1.2. Construct all block var bindings
  int n_vars = block->iter_vars.size();
  for (int i = 0; i < n_vars; ++i) {
    if (!add_remapped_iter_var(i)) {
      print_remapped_iter_var();
      print_single_iter_var(i);
    }
  }
  print_remapped_iter_var();

  // Step 2. Handle block predicate
  if (realize) {
    MXCHECK(realize->predicate.defined() && realize->predicate->dtype.is_bool());
    if (!ir::is_one(realize->predicate)) {
      (*frame)->stmts.push_back(ExprStmtDoc(
          Dialect(d, "where")
              ->Call({d->AsDoc<ExprDoc>(realize->predicate, realize_p->Attr("predicate"))})));
    }
  }
  // Step 3. Handle block read/write regions
  {
    Array<ExprDoc> reads;
    for (int i = 0, n = block->reads.size(); i < n; ++i) {
      reads.push_back(d->AsDoc<ExprDoc>(block->reads[i], block_p->Attr("reads")->ArrayIndex(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(Dialect(d, "reads")->Call(reads)));
    Array<ExprDoc> writes;
    for (int i = 0, n = block->writes.size(); i < n; ++i) {
      writes.push_back(d->AsDoc<ExprDoc>(block->writes[i], block_p->Attr("writes")->ArrayIndex(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(Dialect(d, "writes")->Call(writes)));
  }
  // Step 4. Handle block attributes
  if (!block->annotations.empty()) {
    (*frame)->stmts.push_back(ExprStmtDoc(
        Dialect(d, "block_attr")
            ->Call({d->AsDoc<ExprDoc>(block->annotations, block_p->Attr("annotations"))})));
  }
  // Step 5. Handle `alloc_buffer`
  for (int i = 0, n = block->alloc_buffers.size(); i < n; ++i) {
    ir::Buffer buffer = block->alloc_buffers[i];
    ObjectPath buffer_p = block_p->Attr("alloc_buffers")->ArrayIndex(i);
    IdDoc lhs = DefineBuffer(buffer, *frame, d);
    ExprDoc rhs = BufferDecl(buffer, "alloc_buffer", {}, buffer_p, *frame, d);
    (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
  }
  // Step 6. Handle `match_buffer`
  for (int i = 0, n = block->match_buffers.size(); i < n; ++i) {
    ir::MatchBufferRegion buffer_region = block->match_buffers[i];
    ObjectPath buffer_region_p = block_p->Attr("match_buffers")->ArrayIndex(i);
    StmtDoc doc = d->AsDoc<StmtDoc>(buffer_region, buffer_region_p);
    (*frame)->stmts.push_back(doc);
  }
  // Step 7. Handle init block
  if (block->init.defined()) {
    ir::Stmt init = block->init.value();
    With<IRFrame> init_frame(d, init);
    AsDocBody(init, block_p->Attr("init"), init_frame->get(), d);
    (*frame)->stmts.push_back(
        ScopeDoc(NullOpt, Dialect(d, "init")->Call({}), (*init_frame)->stmts));
  }
  // Step 8. Handle block body
  AsDocBody(block->body, block_p->Attr("body"), frame->get(), d);
  Array<StringRef> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  if (!realize) {
    kwargs_keys.push_back("no_realize");
    kwargs_values.push_back(LiteralDoc::Boolean(true, NullOpt));
  }
  return ScopeDoc(NullOpt,
                  Dialect(d, "block")  //
                      ->Call({LiteralDoc::Str(block->name_hint, block_p->Attr("name_hint"))},
                             kwargs_keys,
                             kwargs_values),
                  (*frame)->stmts);
}

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::ComputeBlockRealize>(
        "", [](ir::ComputeBlockRealize realize, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = PrintBlock(d, realize->block, p->Attr("block"), realize, p);
          // since we do not have d->AsDoc for realize->block,
          // we should add possible doc decoration manually.
          AddDocDecoration<ScopeDoc>(doc, realize->block, p->Attr("block"), d->cfg);
          return doc;
        });

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::ComputeBlock>("",
                                    [](ir::ComputeBlock block, ObjectPath p, IRDocsifier d) -> Doc {
                                      return PrintBlock(d, block, p, NullOpt, NullOpt);
                                    });

}  // namespace ir
}  // namespace matxscript
