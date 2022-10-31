// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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
#include <matxscript/ir/range.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/prim_ops.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace runtime;

Range::Range(PrimExpr begin, PrimExpr end)
    : Range(make_object<RangeNode>(begin, is_zero(begin) ? end : sub(end, begin))) {
}

Range Range::FromMinExtent(PrimExpr min, PrimExpr extent) {
  return Range(make_object<RangeNode>(min, extent));
}

MATXSCRIPT_REGISTER_GLOBAL("ir.Range_from_min_extent").set_body_typed(Range::FromMinExtent);

MATXSCRIPT_REGISTER_GLOBAL("ir.Range").set_body([](PyArgs args) -> RTValue {
  return Range(args[0].As<PrimExpr>(), args[1].As<PrimExpr>());
});

MATXSCRIPT_REGISTER_NODE_TYPE(RangeNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RangeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RangeNode*>(node.get());
      p->stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
    });

}  // namespace ir
}  // namespace matxscript
