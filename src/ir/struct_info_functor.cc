/*
 * Taken from https://github.com/apache/tvm/blob/unity/src/relax/ir/struct_info_functor.cc
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
 * \file struct_info_functor.cc
 * \brief Implementations of struct info functors.
 */
#include <matxscript/ir/struct_info_functor.h>

namespace matxscript {
namespace ir {

void StructInfoVisitor::VisitStructInfo_(const ObjectStructInfoNode* op) {
}

void StructInfoVisitor::VisitStructInfo_(const PrimStructInfoNode* op) {
}

void StructInfoVisitor::VisitStructInfo_(const ShapeStructInfoNode* op) {
  if (op->values.defined()) {
    for (PrimExpr value : op->values.value()) {
      this->VisitStructInfoExprField(value);
    }
  }
}

void StructInfoVisitor::VisitStructInfo_(const TensorStructInfoNode* op) {
  if (op->shape.defined()) {
    this->VisitStructInfoExprField(op->shape.value());
  }
}

void StructInfoVisitor::VisitStructInfo_(const TupleStructInfoNode* op) {
  for (StructInfo field : op->fields) {
    this->VisitStructInfo(field);
  }
}

StructInfo StructInfoMutator::VisitStructInfo_(const ObjectStructInfoNode* op) {
  return runtime::GetRef<StructInfo>(op);
}

StructInfo StructInfoMutator::VisitStructInfo_(const PrimStructInfoNode* op) {
  return runtime::GetRef<StructInfo>(op);
}

StructInfo StructInfoMutator::VisitStructInfo_(const ShapeStructInfoNode* op) {
  Optional<Array<PrimExpr>> values;

  if (op->values.defined()) {
    // if no changes are made the original array will be returned.
    values = op->values.value().Map(
        [this](const PrimExpr& expr) { return this->VisitStructInfoExprField(expr); });
  }

  if (values.same_as(op->values)) {
    return runtime::GetRef<StructInfo>(op);
  } else {
    return ShapeStructInfo(values.value(), op->span);
  }
}

StructInfo StructInfoMutator::VisitStructInfo_(const TensorStructInfoNode* op) {
  Optional<HLOExpr> shape;

  if (op->shape.defined()) {
    shape = this->VisitStructInfoExprField(op->shape.value());
  }

  if (shape.same_as(op->shape)) {
    return runtime::GetRef<StructInfo>(op);
  } else {
    return TensorStructInfo(shape.value(), op->dtype, op->span);
  }
}

StructInfo StructInfoMutator::VisitStructInfo_(const TupleStructInfoNode* op) {
  Array<StructInfo> fields =
      op->fields.Map([this](const StructInfo& sinfo) { return this->VisitStructInfo(sinfo); });

  if (fields.same_as(op->fields)) {
    return runtime::GetRef<StructInfo>(op);
  } else {
    return TupleStructInfo(fields, op->span);
  }
}

}  // namespace ir
}  // namespace matxscript
