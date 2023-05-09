/*
 * Taken from https://github.com/apache/tvm/blob/unity/src/relax/analysis/struct_info_analysis.cc
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
 * \file struct_info_analysis.cc
 * \brief Implementations of foundation struct info analysis
 *
 * \note Update this file when you added a new StructInfo.
 */
#include <matxscript/ir/analysis.h>

#include <matxscript/ir/expr.h>
#include <matxscript/ir/struct_info.h>
#include <matxscript/ir/struct_info_functor.h>

namespace matxscript {
namespace ir {

//--------------------------
// GetStaticType
//--------------------------
class StaticTypeDeriver : public StructInfoFunctor<Type(const StructInfo&)> {
 public:
  Type VisitStructInfo_(const ObjectStructInfoNode* op) final {
    return ObjectType(false, op->span);
  }

  Type VisitStructInfo_(const PrimStructInfoNode* op) final {
    return PrimType(op->dtype);
  }

  Type VisitStructInfo_(const ShapeStructInfoNode* op) final {
    return ShapeType(op->ndim, op->span);
  }

  Type VisitStructInfo_(const TensorStructInfoNode* op) final {
    return DynTensorType(op->ndim, op->dtype);
  }

  Type VisitStructInfo_(const TupleStructInfoNode* op) final {
    Array<Type> fields = op->fields.Map(
        [this](const StructInfo& sinfo) -> Type { return this->VisitStructInfo(sinfo); });
    return TupleType(fields, op->span);
  }
};

Type GetStaticType(const StructInfo& info) {
  return StaticTypeDeriver()(info);
}

MATXSCRIPT_REGISTER_GLOBAL("relax.analysis.GetStaticType")
    .set_body_typed([](const StructInfo& info) { return GetStaticType(info); });

//--------------------------
// StructInfoFromType
//--------------------------

StructInfo StructInfoFromType(const Type& type) {
  if (type.as<ObjectTypeNode>()) {
    return ObjectStructInfo(type->span);
  } else if (const PrimTypeNode* prim_type = type.as<PrimTypeNode>()) {
    return PrimStructInfo(prim_type->dtype, prim_type->span);
  } else if (const ShapeTypeNode* shape_type = type.as<ShapeTypeNode>()) {
    return ShapeStructInfo(shape_type->ndim, type->span);
  } else if (const DynTensorTypeNode* tensor_type = type.as<DynTensorTypeNode>()) {
    return TensorStructInfo(tensor_type->dtype, tensor_type->ndim);
  } else if (const TupleTypeNode* tuple_type = type.as<TupleTypeNode>()) {
    Array<StructInfo> fields;
    for (const Type& field : tuple_type->fields) {
      fields.push_back(StructInfoFromType(field));
    }
    return TupleStructInfo(fields, type->span);
  } else if (const FuncTypeNode* func_type = type.as<FuncTypeNode>()) {
    MXLOG(FATAL) << "Unsupported type: " << type;
    return StructInfo();
  } else {
    MXLOG(FATAL) << "Unsupported type: " << type;
    return StructInfo();
  }
}

}  // namespace ir
}  // namespace matxscript
