/*
 * Taken from https://github.com/apache/tvm/blob/unity/include/tvm/relax/struct_info.h
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once

#include <matxscript/ir/base.h>

namespace matxscript {
namespace ir {

/*!
 * \brief Opaque object.
 */
class ObjectStructInfoNode : public StructInfoNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ObjectStructInfoNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(0);
  }

  static constexpr const char* _type_key = "ir.ObjectStructInfo";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ObjectStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to ObjectStructInfoNode.
 * \sa ObjectStructInfoNode
 */
class ObjectStructInfo : public StructInfo {
 public:
  MATX_DLL ObjectStructInfo(Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ObjectStructInfo,
                                                   StructInfo,
                                                   ObjectStructInfoNode);
};

/*!
 * \brief Primitive value.
 */
class PrimStructInfoNode : public StructInfoNode {
 public:
  /*! \brief Underlying data type of the primitive value */
  runtime::DataType dtype;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const PrimStructInfoNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
  }

  static constexpr const char* _type_key = "ir.PrimStructInfo";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to PrimStructInfoNode.
 * \sa PrimStructInfoNode
 */
class PrimStructInfo : public StructInfo {
 public:
  MATX_DLL PrimStructInfo(runtime::DataType dtype, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PrimStructInfo, StructInfo, PrimStructInfoNode);
};

/*!
 * \brief StructInfo of shape value.
 */
class ShapeStructInfoNode : public StructInfoNode {
 public:
  /*! \brief optionally stores the symbolic value patterns of the shape */
  Optional<Array<PrimExpr>> values;
  /*!
   * \brief The number of dimension of the shape, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim;

  /*! \return Whether the struct info contains unknown ndim. */
  bool IsUnknownNdim() const {
    return ndim == kUnknownNDim;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("values", &values);
    v->Visit("ndim", &ndim);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ShapeStructInfoNode* other, SEqualReducer equal) const {
    return equal(values, other->values) && equal(ndim, other->ndim);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(values);
    hash_reduce(ndim);
  }

  static constexpr const char* _type_key = "ir.ShapeStructInfo";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ShapeStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to ShapeStructInfoNode.
 * \sa ShapeStructInfoNode
 */
class ShapeStructInfo : public StructInfo {
 public:
  /*!
   * \brief Construction with known symbolic shape patterns
   * \param values The symbolic shape values
   * \param span The span of the AST.
   */
  MATX_DLL ShapeStructInfo(Array<PrimExpr> values, Span span = Span());
  /*!
   * \brief Construction with known unknown symbolic shape patterns.
   * \param ndim Number of dimensions -- can be kUnknownNDim
   * \param span The span of the AST.
   */
  MATX_DLL ShapeStructInfo(int ndim, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ShapeStructInfo,
                                                   StructInfo,
                                                   ShapeStructInfoNode);
};

/*!
 * \brief StructInfo of Tensor.
 */
class TensorStructInfoNode : public StructInfoNode {
 public:
  /*!
   * \brief optionally store the shape expression of the tensor.
   * \note shape must be normalized: it can only be NullOpt or ShapeExpr or Var.
   */
  Optional<HLOExpr> shape;
  /*! \brief The content data type, use void to denote the dtype is unknown. */
  runtime::DataType dtype;
  /*!
   * \brief The number of dimension of the tensor, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim;

  /*! \return Whether the struct info contains unknown ndim. */
  bool IsUnknownNdim() const {
    return ndim == kUnknownNDim;
  }

  /*! \return Whether the struct info contains unknown dtype. */
  bool IsUnknownDtype() const {
    return dtype.is_void();
  }

  /*! \return Shape if it is known. */
  Optional<Array<PrimExpr>> GetShape() const {
    if (!shape.defined())
      return {};
    ShapeStructInfo shape_sinfo =
        runtime::Downcast<ShapeStructInfo>(this->shape.value()->struct_info_);
    return shape_sinfo->values;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("ndim", &ndim);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TensorStructInfoNode* other, SEqualReducer equal) const {
    return equal(shape, other->shape) && equal(ndim, other->ndim) && equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(shape);
    hash_reduce(dtype);
    hash_reduce(ndim);
  }

  static constexpr const char* _type_key = "ir.TensorStructInfo";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TensorStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to TensorStructInfoNode.
 * \sa TensorStructInfoNode
 */
class TensorStructInfo : public StructInfo {
 public:
  /*!
   * \brief Construction with a known shape expression.
   * \param shape The shape of the tensor.
   * \param dtype The data type of tensor's elements.
   * \param span The span of the AST.
   *
   * \note shape must already be normalized.
   */
  MATX_DLL TensorStructInfo(HLOExpr shape, runtime::DataType dtype, Span span = Span());

  /*!
   * \brief Construction with an unknown shape expression.
   * \param dtype The data type of tensor's elements.
   * \param ndim The number of dimensions
   * \param span The span of the AST.
   */
  MATX_DLL TensorStructInfo(runtime::DataType dtype, int ndim, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorStructInfo,
                                                   StructInfo,
                                                   TensorStructInfoNode);
};

/*!
 * \brief StructInfo of Tuple.
 */
class TupleStructInfoNode : public StructInfoNode {
 public:
  /*! \brief The struct info of tuple fields. */
  Array<StructInfo> fields;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TupleStructInfoNode* other, SEqualReducer equal) const {
    return equal(fields, other->fields);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(fields);
  }

  static constexpr const char* _type_key = "ir.TupleStructInfo";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TupleStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to TupleStructInfoNode.
 * \sa TupleStructInfoNode
 */
class TupleStructInfo : public StructInfo {
 public:
  /*!
   * \brief Constructor
   * \param fields Struct info of tuple fields.
   * \param span The span of the AST.
   */
  MATX_DLL TupleStructInfo(Array<StructInfo> fields, Span span = Span());

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TupleStructInfo,
                                                   StructInfo,
                                                   TupleStructInfoNode);
};

/*!
 * \brief Match and check if expr have StructInfo T and return it.
 *
 * \param expr The input expression.
 * \return The result of match.
 * \tparam T the underlying structure info type
 */
template <typename T>
inline Optional<T> MatchStructInfo(const HLOExpr& expr) {
  using TNode = typename T::ContainerType;
  if (const TNode* ptr = expr->struct_info_.as<TNode>()) {
    return runtime::GetRef<T>(ptr);
  } else {
    return NullOpt;
  }
}

/*!
 * \brief Get the structure info of a given expr and try to cast it as const T*.
 *
 * \param expr The input expression.
 * \return The pointer. Returns nullptr if the type does not match
 * \tparam T the underlying structure info type
 */
template <typename T>
inline const T* GetStructInfoAs(const HLOExpr& expr) {
  MXCHECK(expr->struct_info_.defined())
      << "The struct_info is not populated, check if you have normalized the expr";
  return expr->struct_info_.as<T>();
}

/*!
 * \brief Get the underlying structure info of expr.
 *
 * \param expr The input expression.
 * \return underlying struct info.
 */
inline StructInfo GetStructInfo(const HLOExpr& expr) {
  auto* ptr = expr->struct_info_.as<StructInfoNode>();
  MXCHECK(ptr) << "The struct_info is not populated, check if you have normalized the expr";
  return runtime::GetRef<StructInfo>(ptr);
}

/*!
 * \brief Whether the expr has void struct info.
 *
 * \param expr The input expression.
 * \return Whether the expr has void struct info.
 */
inline bool HasVoidStructInfo(const HLOExpr& expr) {
  auto* ptr = expr->struct_info_.as<TupleStructInfoNode>();
  return ptr != nullptr && ptr->fields.size() == 0;
}

/*!
 * \brief Update the struct info of an HLOExpr.
 * \param expr The HLOExpr whose struct info to be updated.
 * \param struct_info The struct_info assigned.
 * \note We ensure idempotence, that is we can only update the struct_info of an HLOExpr only
 *  if the original one is nullptr.
 */
MATX_DLL void UpdateStructInfo(HLOExpr expr, StructInfo struct_info);

}  // namespace ir
}  // namespace matxscript
