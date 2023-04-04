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
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/type_helper_macros.h>

namespace matxscript {
namespace ir {

using namespace runtime;
using namespace ::matxscript::ir::printer;

RangeExpr::RangeExpr(PrimExpr start, PrimExpr stop, PrimExpr step, Span span) {
  auto n = make_object<RangeExprNode>();
  n->start = std::move(start);
  n->stop = std::move(stop);
  n->step = std::move(step);
  n->checked_type_ = RangeType(span);
  n->span = std::move(span);
  data_ = std::move(n);
}

MATXSCRIPT_REGISTER_GLOBAL("ir.RangeExpr")
    .set_body_typed([](PrimExpr start, PrimExpr stop, PrimExpr step, Span span) -> RTValue {
      return RangeExpr(std::move(start), std::move(stop), std::move(step), std::move(span));
    });

MATXSCRIPT_REGISTER_NODE_TYPE(RangeExprNode);

MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<RangeExpr>("", [](RangeExpr r, ObjectPath p, IRDocsifier d) -> Doc {
      // TODO: optimize range args
      ExprDoc start = d->AsDoc<ExprDoc>(r->start, p->Attr("start"));
      ExprDoc stop = d->AsDoc<ExprDoc>(r->stop, p->Attr("stop"));
      ExprDoc step = d->AsDoc<ExprDoc>(r->step, p->Attr("step"));
      return IdDoc("range")->Call({start, stop, step});
    });

}  // namespace ir
}  // namespace matxscript
