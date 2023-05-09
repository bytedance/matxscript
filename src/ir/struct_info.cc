/*
 * Taken from https://github.com/apache/tvm/blob/unity/src/relax/ir/struct_info.cc
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
 * \file src/ir/struct_info.cc
 * \brief struct info.
 */
#include <matxscript/ir/struct_info.h>

#include <matxscript/ir/analysis.h>
#include <matxscript/ir/hlo_expr.h>
#include <matxscript/ir/hlo_var.h>
#include <matxscript/ir/prim_expr.h>
#include <matxscript/ir/prim_ops.h>
#include <matxscript/ir/prim_var.h>
#include <matxscript/ir/struct_info_functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

ObjectStructInfo::ObjectStructInfo(Span span) {
  ObjectPtr<ObjectStructInfoNode> n = runtime::make_object<ObjectStructInfoNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ObjectStructInfoNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ObjectStructInfo").set_body_typed([](Span span) {
  return ObjectStructInfo(span);
});

// Prim
PrimStructInfo::PrimStructInfo(runtime::DataType dtype, Span span) {
  ObjectPtr<PrimStructInfoNode> n = runtime::make_object<PrimStructInfoNode>();
  n->dtype = dtype;
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(PrimStructInfoNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.PrimStructInfo")
    .set_body_typed([](runtime::DataType dtype, Span span) { return PrimStructInfo(dtype, span); });

// Shape
ShapeStructInfo::ShapeStructInfo(Array<PrimExpr> values, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = runtime::make_object<ShapeStructInfoNode>();
  n->ndim = static_cast<int>(values.size());
  n->values = values.Map([](PrimExpr value) -> PrimExpr {
    if (value->IsInstance<IntImmNode>()) {
      return cast(runtime::DataType::Int(64), value, value->span);
    }
    MXCHECK(value.dtype() == runtime::DataType::Int(64))
        << "the value in ShapeStructInfo can only have dtype of int64";
    return value;
  });
  n->span = std::move(span);
  data_ = std::move(n);
}

ShapeStructInfo::ShapeStructInfo(int ndim, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = runtime::make_object<ShapeStructInfoNode>();
  MXCHECK_GE(ndim, -1) << "ndim of ShapeStructInfo must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(ShapeStructInfoNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.ShapeStructInfo")
    .set_body_typed([](Optional<Array<PrimExpr>> values, int ndim, Span span) {
      if (values.defined()) {
        MXCHECK_EQ(ndim, kUnknownNDim) << "ValueError: Cannot both specify values and ndim";
        return ShapeStructInfo(values.value(), span);
      } else {
        return ShapeStructInfo(ndim, span);
      }
    });

// Tensor
TensorStructInfo::TensorStructInfo(HLOExpr shape, runtime::DataType dtype, Span span) {
  ObjectPtr<TensorStructInfoNode> n = runtime::make_object<TensorStructInfoNode>();
  // assign ndim before move
  Optional<ShapeStructInfo> sinfo = MatchStructInfo<ShapeStructInfo>(shape);
  MXCHECK(sinfo) << "We expect shape to contain pre-set shape struct info";
  MXCHECK(shape.defined()) << "Must provide a shape in this constructor";
  MXCHECK(shape->IsInstance<ShapeExprNode>() || shape->IsInstance<HLOVarNode>())
      << "We require shape to be normalized when constructing TensorStructInfo";
  n->ndim = sinfo.get()->ndim;
  // assign rest of the fields.
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->span = std::move(span);
  data_ = std::move(n);
}

TensorStructInfo::TensorStructInfo(runtime::DataType dtype, int ndim, Span span) {
  ObjectPtr<TensorStructInfoNode> n = runtime::make_object<TensorStructInfoNode>();
  MXCHECK_GE(ndim, -1) << "ndim of TensorStructInfo must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->dtype = dtype;
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(TensorStructInfoNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.TensorStructInfo")
    .set_body_typed([](Optional<HLOExpr> shape, runtime::DataType dtype, int ndim, Span span) {
      if (shape.defined()) {
        MXCHECK_EQ(ndim, kUnknownNDim) << "ValueError: Cannot both specify shape and ndim";
        return TensorStructInfo(shape.value(), dtype, span);
      } else {
        return TensorStructInfo(dtype, ndim, span);
      }
    });

// Tuple
TupleStructInfo::TupleStructInfo(Array<StructInfo> fields, Span span) {
  ObjectPtr<TupleStructInfoNode> n = runtime::make_object<TupleStructInfoNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_NODE_TYPE(TupleStructInfoNode);

MATXSCRIPT_REGISTER_GLOBAL("ir.TupleStructInfo")
    .set_body_typed([](Array<StructInfo> fields, Span span) {
      return TupleStructInfo(fields, span);
    });

// Helper functions
void UpdateStructInfo(HLOExpr expr, StructInfo struct_info) {
  MXCHECK(!expr->struct_info_.defined())
      << "the struct_info_ of the Expr to be updated must be nullptr for idempotency";
  expr->struct_info_ = struct_info;
  // also set checked type
  expr->checked_type_ = GetStaticType(struct_info);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.UpdateStructInfo")
    .set_body_typed([](HLOExpr expr, StructInfo struct_info) {
      UpdateStructInfo(expr, struct_info);
    });

MATXSCRIPT_REGISTER_GLOBAL("ir.ExprStructInfo").set_body_typed([](HLOExpr expr) {
  return GetStructInfo(expr);
});

}  // namespace ir
}  // namespace matxscript
