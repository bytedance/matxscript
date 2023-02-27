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
#include <matxscript/ir/range_expr.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/type_helper_macros.h>

namespace matxscript {
namespace ir {

using namespace runtime;

RangeExpr::RangeExpr(PrimExpr start, PrimExpr stop, PrimExpr step, Span span) {
  auto n = make_object<RangeExprNode>();
  n->start = std::move(start);
  n->stop = std::move(stop);
  n->step = std::move(step);
  n->checked_type_ = RangeType(span);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.RangeExpr").set_body([](PyArgs args) -> RTValue {
  return RangeExpr(MATXSCRIPT_TYPE_AS(args[0], PrimExpr),
                   MATXSCRIPT_TYPE_AS(args[1], PrimExpr),
                   MATXSCRIPT_TYPE_AS(args[2], PrimExpr));
});

MATXSCRIPT_REGISTER_NODE_TYPE(RangeExprNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RangeExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RangeExprNode*>(node.get());
      p->stream << "range(start=" << op->start << ", stop=" << op->stop << ", step=" << op->step
                << ')';
    });

}  // namespace ir
}  // namespace matxscript
