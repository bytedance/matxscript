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
 * \file matx/ir/tensor_stmt.h
 * \brief ir.map_block.
 */
#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <matxscript/ir/_base/optional_ref.h>
#include <matxscript/ir/base.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/ir/prim_var.h>
#include <matxscript/ir/range_expr.h>

namespace matxscript {
namespace ir {

/******************************************************************************
 * Buffer
 *****************************************************************************/

/*! \brief buffer type */
enum BufferType : int {
  kDefault = 1,
  // Maps buffer[i][j][k] -> buffer[i][0][k] if dimension i's shape equals 1.
  kAutoBroadcast = 2,
};

/*! \brief Node to represent a buffer */
class BufferNode : public Object {
 public:
  // Data fields.
  /*!
   * \brief The pointer to the head of the data
   * \sa data_alignment The alignment of data in bytes.
   */
  PrimVar data;
  /*! \brief data type in the content of the tensor */
  runtime::DataType dtype;
  /*! \brief The type of the buffer prior to flattening
   *
   * This contains the shape as it is accessed by
   * BufferLoad/BufferStore nodes, and used by the low-level code
   * generators.
   */
  Array<PrimExpr> shape;
  /*!
   * \brief Separators between input axes when generating flattened output axes
   *
   * For buffers representing flat 1-d memory (e.g. any buffer in
   * RAM), this should be an empty array.  For buffers representing
   * non-flat memory, each entry in axis_separators should be the
   * first input axis that is part of a new flattened axis.
   */
  Array<IntImm> axis_separators;
  /*!
   * \brief The strides of each dimension
   *  This can be an empty array, indicating array is contiguous
   */
  Array<PrimExpr> strides;
  /*! \brief The offset in terms of number of dtype elements (including lanes) */
  PrimExpr elem_offset;
  // Meta data
  /*! \brief optional name of the buffer */
  StringRef name;
  /*! \brief Alignment requirement of data pointer in bytes. */
  int data_alignment;
  /*!
   * \brief Factor of elem_offset field,
   *  elem_offset is guaranteed to be multiple of offset_factor.
   */
  int offset_factor;
  /*! \brief buffer type */
  BufferType buffer_type;

  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  /*! \brief constructor */
  BufferNode() {
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
    v->Visit("strides", &strides);
    v->Visit("axis_separators", &axis_separators);
    v->Visit("elem_offset", &elem_offset);
    v->Visit("name", &name);
    v->Visit("data_alignment", &data_alignment);
    v->Visit("offset_factor", &offset_factor);
    v->Visit("buffer_type", &buffer_type);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const BufferNode* other, SEqualReducer equal) const {
    // Use DefEqual as buffer can define variables in its semantics,
    // skip name as name is not important.
    return equal.DefEqual(data, other->data) && equal(dtype, other->dtype) &&
           equal.DefEqual(shape, other->shape) && equal.DefEqual(strides, other->strides) &&
           equal.DefEqual(axis_separators, other->axis_separators) &&
           equal.DefEqual(elem_offset, other->elem_offset) &&
           equal(data_alignment, other->data_alignment) && equal(buffer_type, other->buffer_type);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(data);
    hash_reduce(dtype);
    hash_reduce.DefHash(shape);
    hash_reduce.DefHash(strides);
    hash_reduce.DefHash(elem_offset);
    hash_reduce.DefHash(axis_separators);
    hash_reduce(data_alignment);
    hash_reduce(buffer_type);
  }
  /*! \return preferred index type for this buffer node */
  runtime::DataType DefaultIndexType() const {
    return shape.size() != 0 ? shape[0].dtype() : runtime::DataType::Int(32);
  }

  static constexpr const char* _type_key = "ir.Buffer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(BufferNode, Object);
};

/*!
 * \brief Buffer is a symbolic n-darray structure.
 *  It is a composition of primitive symbolic types,
 *  used to specify the memory layout of the Tensor used in program input.
 */
class Buffer : public ObjectRef {
 public:
  // User can specify data_alignment and offset_factor to be 0
  // A default value will be picked.
  MATX_DLL Buffer(PrimVar data,
                  runtime::DataType dtype,
                  Array<PrimExpr> shape,
                  Array<PrimExpr> strides,
                  PrimExpr elem_offset,
                  StringRef name,
                  int data_alignment,
                  int offset_factor,
                  BufferType buffer_type,
                  Array<IntImm> axis_separators = {},
                  Span span = Span());

  /*!
   * \brief Create an Expr that does a vector load at begin index.
   * \param begin The beginning index
   * \param dtype The data type to be loaded.
   */
  MATX_DLL PrimExpr vload(Array<PrimExpr> begin, runtime::DataType dtype) const;
  /*!
   * \brief Create a Stmt that does a vector store at begin index.
   * \param begin The beginning index
   * \param value The value to be stored.
   */
  MATX_DLL Stmt vstore(Array<PrimExpr> begin, PrimExpr value) const;

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Buffer, ObjectRef, BufferNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(BufferNode);
};

/******************************************************************************
 * BufferRegion
 *****************************************************************************/

/*!
 * \brief Representing the region of multi-dimensional buffer access.
 */
class BufferRegionNode : public Object {
 public:
  /*! \brief The buffer of the buffer region. */
  Buffer buffer;
  /*! \brief The region array of the buffer region. */
  Array<RangeExpr> region;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("region", &region);
  }

  bool SEqualReduce(const BufferRegionNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(region, other->region);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(region);
  }

  static constexpr const char* _type_key = "ir.BufferRegion";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(BufferRegionNode, Object);
};

/*!
 * \brief Managed reference to BufferRegionNode.
 * \sa BufferRegionNode
 */
class BufferRegion : public ObjectRef {
 public:
  MATX_DLL explicit BufferRegion(Buffer buffer, Array<RangeExpr> region);

  /*!
   * \brief Create a BufferRegion which is full region of the given buffer.
   * \param buffer The buffer to generate full BufferRegion.
   * \return The BufferRegion which covers all region of the given buffer
   */
  MATX_DLL static BufferRegion FullRegion(Buffer buffer);

  /*!
   * \brief Create a BufferRegion which is a single point of the given buffer.
   * \param buffer The buffer to generate single point BufferRegion.
   * \param indices The access point indices of the buffer
   * \return The BufferRegion which is the single point of the given buffer.
   */
  MATX_DLL static BufferRegion FromPoint(Buffer buffer, Array<PrimExpr> indices);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(BufferRegion, ObjectRef, BufferRegionNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(BufferRegionNode);
};

/******************************************************************************
 * MatchBufferRegion
 *****************************************************************************/

/*!
 * \brief Match introduces a constraint that the source buffer region can be remapped to the data
 * layout specified by the buffer field. The constraint can be checked in later part of lowering (or
 * optionally during runtime).
 *
 * MatchBufferRegion provides a mechanism to represent data layout and compactness constraints in
 * low-level hardware primitives in the IR and defer the check after the sequence of
 * transformations.
 */
class MatchBufferRegionNode : public Object {
 public:
  /*! \brief The target buffer. */
  Buffer buffer;
  /*! \brief The source buffer region. */
  BufferRegion source;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("source", &source);
  }

  bool SEqualReduce(const MatchBufferRegionNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(source, other->source);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(source);
  }

  static constexpr const char* _type_key = "ir.MatchBufferRegion";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(MatchBufferRegionNode, Object);
};

/*!
 * \brief Managed reference to MatchBufferRegionNode.
 * \sa MatchBufferRegionNode
 */
class MatchBufferRegion : public ObjectRef {
 public:
  MATX_DLL explicit MatchBufferRegion(Buffer buffer, BufferRegion source);

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(MatchBufferRegion, ObjectRef, MatchBufferRegionNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(MatchBufferRegionNode);
};

/******************************************************************************
 * BufferLoad
 *****************************************************************************/

/*!
 * \brief Load value from the high dimension buffer.
 *
 * \code
 *
 *  value = buffer[i, j];
 *
 * \endcode
 * \sa BufferLoad
 */
class BufferLoadNode : public PrimExprNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The indices location to be loaded. */
  Array<PrimExpr> indices;

  void VisitAttrs(AttrVisitor* v) {
    PrimExprNode::VisitAttrs(v);
    v->Visit("buffer", &buffer);
    v->Visit("indices", &indices);
  }

  bool SEqualReduce(const BufferLoadNode* other, SEqualReducer equal) const {
    return PrimExprNode::SEqualReduce(other, equal) && equal(buffer, other->buffer) &&
           equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    PrimExprNode::SHashReduce(hash_reduce);
    hash_reduce(buffer);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "ir.BufferLoad";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(BufferLoadNode, PrimExprNode);

 private:
  /*! \brief Set the dtype based on the buffer/indices
   *
   * Usually, the BufferLoad's dtype will be the same dtype as the
   * buffer.  This may have a different number of lanes than the
   * buffer's dtype if index values have more than 1 lane.
   *
   * This function should only be called during construction and after
   * CopyOnWrite.  Friend class used here to restrict usage.
   */
  void LegalizeDType();
  friend class BufferLoad;
};

/*!
 * \brief Managed reference to BufferLoadNode.
 * \sa BufferLoadNode
 */
class BufferLoad : public PrimExpr {
 public:
  MATX_DLL explicit BufferLoad(Buffer buffer, Array<PrimExpr> indices, Span span = Span());
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(BufferLoad, PrimExpr, BufferLoadNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(BufferLoadNode);
};

/******************************************************************************
 * BufferStore
 *****************************************************************************/

/*!
 * \brief Store value to the high dimension buffer.
 *
 * \code
 *
 *  buffer[i, j] = value;
 *
 * \endcode
 * \sa BufferStore
 */
class BufferStoreNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The value to be stored. */
  PrimExpr value;
  /*! \brief The indices location to be stored. */
  Array<PrimExpr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const BufferStoreNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(value, other->value) &&
           equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(value);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "ir.BufferStore";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(BufferStoreNode, StmtNode);
};

/*!
 * \brief Managed reference to BufferStoreNode.
 * \sa BufferStoreNode
 */
class BufferStore : public Stmt {
 public:
  MATX_DLL explicit BufferStore(Buffer buffer,
                                PrimExpr value,
                                Array<PrimExpr> indices,
                                Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(BufferStore, Stmt, BufferStoreNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(BufferStoreNode);
};

/******************************************************************************
 * ComputeBlock
 *****************************************************************************/

/*!
 * \brief A map block is a basic schedule unit in kernel ir.
 * \note Block's body is parameterized by iter vars.
 * \code
 *
 *  for i, j, k in zip(range(M), range(N), range(K)):
 *      a[i][j][k] = 1
 *
 * \endcode
 */
class ComputeBlockNode : public StmtNode {
 public:
  /*! \brief The variables of the block. */
  Array<PrimIterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  Array<BufferRegion> reads;
  /*! \brief The write buffer regions of the block. */
  Array<BufferRegion> writes;
  /*! \brief The name_hint of the block. */
  StringRef name_hint;
  /*! \brief The body of the block. */
  Stmt body;
  /*!
   * \brief The init statement is executed during the first iteration of reduction loops in a
   *  reduction block. The optional init field allows us to represent initialization and
   *  reduction update in a single block and transform them collectively.
   *  We also provide primitives to decompose the init into a separate block during scheduling.
   *  Init field is `NullOpt` if there is no reduction iter_vars
   */
  Optional<Stmt> init;
  /*! \brief The buffer allocated in the block. */
  Array<Buffer> alloc_buffers;
  /*! \brief The match buffer regions. */
  Array<MatchBufferRegion> match_buffers;
  /*! \brief The annotation of the block. */
  Map<StringRef, ObjectRef> annotations;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("iter_vars", &iter_vars);
    v->Visit("reads", &reads);
    v->Visit("writes", &writes);
    v->Visit("name_hint", &name_hint);
    v->Visit("body", &body);
    v->Visit("init", &init);
    v->Visit("alloc_buffers", &alloc_buffers);
    v->Visit("match_buffers", &match_buffers);
    v->Visit("annotations", &annotations);
  }

  bool SEqualReduce(const ComputeBlockNode* other, SEqualReducer equal) const {
    // Need first reduce iter_vars, alloc_buffers and match_buffers to define new vars
    return equal.DefEqual(iter_vars, other->iter_vars) &&
           equal(alloc_buffers, other->alloc_buffers) &&
           equal(match_buffers, other->match_buffers) && equal(reads, other->reads) &&
           equal(writes, other->writes) && equal(body, other->body) && equal(init, other->init) &&
           equal(annotations, other->annotations);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(iter_vars);
    hash_reduce(alloc_buffers);
    hash_reduce(match_buffers);
    hash_reduce(reads);
    hash_reduce(writes);
    hash_reduce(body);
    hash_reduce(init);
    hash_reduce(annotations);
  }

  static constexpr const char* _type_key = "ir.ComputeBlock";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ComputeBlockNode, StmtNode);
};

/*!
 * \brief Managed reference to BlockNode.
 * \sa BlockNode
 */
class ComputeBlock : public Stmt {
 public:
  MATX_DLL explicit ComputeBlock(
      Array<PrimIterVar> iter_vars,
      Array<BufferRegion> reads,
      Array<BufferRegion> writes,
      StringRef name_hint,
      Stmt body,
      Optional<Stmt> init = NullOpt,
      Array<Buffer> alloc_buffers = Array<Buffer>(),
      Array<MatchBufferRegion> match_buffers = Array<MatchBufferRegion>(),
      Map<StringRef, ObjectRef> annotations = Map<StringRef, ObjectRef>(),
      Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ComputeBlock, Stmt, ComputeBlockNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(ComputeBlockNode);
};

/*!
 * \brief A block realization node represents execution of the block at the binding values.
 */
class ComputeBlockRealizeNode : public StmtNode {
 public:
  /*! \brief The corresponding values of the iter vars. */
  Array<PrimExpr> iter_values;
  /*!
   * \brief The predicate of the block realization, the block will only be executed when the
   * predicate is true.
   */
  PrimExpr predicate;
  /*! \brief The block to be realized. */
  ComputeBlock block;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("iter_values", &iter_values);
    v->Visit("predicate", &predicate);
    v->Visit("block", &block);
  }

  bool SEqualReduce(const ComputeBlockRealizeNode* other, SEqualReducer equal) const {
    return equal(iter_values, other->iter_values) && equal(predicate, other->predicate) &&
           equal(block, other->block);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(iter_values);
    hash_reduce(predicate);
    hash_reduce(block);
  }

  static constexpr const char* _type_key = "ir.ComputeBlockRealize";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ComputeBlockRealizeNode, StmtNode);
};

/*!
 * \brief Managed reference to ComputeBlockRealizeNode
 * \sa ComputeBlockRealizeNode
 */
class ComputeBlockRealize : public Stmt {
 public:
  MATX_DLL explicit ComputeBlockRealize(Array<PrimExpr> iter_values,
                                        PrimExpr predicate,
                                        ComputeBlock block,
                                        Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(ComputeBlockRealize, Stmt, ComputeBlockRealizeNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(ComputeBlockRealizeNode);
};

}  // namespace ir
}  // namespace matxscript
