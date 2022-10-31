// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by Halide/TVM IR.
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
 * \file matx/ir/hlo_builtin.h
 * \brief high level ir builtin intrinsics.
 *
 */
#pragma once

#include <matxscript/ir/base.h>

namespace matxscript {
namespace ir {

MATX_DLL BaseExpr add(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr sub(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr mul(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr floordiv(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr floormod(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr greater_than(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr greater_or_equal(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr less_than(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr less_or_equal(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr equal(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr not_equal(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr logic_and(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr logic_or(BaseExpr a, BaseExpr b, Span span = Span());
MATX_DLL BaseExpr logic_not(BaseExpr a, Span span = Span());

}  // namespace ir
}  // namespace matxscript
