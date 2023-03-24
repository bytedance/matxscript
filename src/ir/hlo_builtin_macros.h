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
#pragma once
#include <matxscript/ir/op_attr_types.h>
#include <matxscript/ir/op_expr.h>

namespace matxscript {
namespace ir {
namespace builtin {

#define MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC(OpName) \
  const Op& OpName() {                                \
    static const Op& op = Op::Get("ir." #OpName);     \
    return op;                                        \
  }                                                   \
  MATXSCRIPT_IR_REGISTER_OP("ir." #OpName)

/******************************************************************************
 * Method
 *****************************************************************************/

#define MATXSCRIPT_IR_DEFINE_HLO_METHOD(T, OpName, KernelMethod)                      \
  MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC(T##_##OpName)                                 \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque)) \
      .set_attr<TMethodSymbol>("TMethodSymbol", #KernelMethod)                        \
      .set_attr<TPrinterMethodSymbol>("TPrinterMethodSymbol", #OpName)

/******************************************************************************
 * Global Function
 *****************************************************************************/

#define MATXSCRIPT_IR_DEFINE_HLO_MODULE_FUNC(Prefix, OpName)                                   \
  MATXSCRIPT_IR_DEFINE_HLO_BUILTIN_FUNC(Prefix##_##OpName)                                     \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))          \
      .set_attr<TGlobalSymbol>("TGlobalSymbol", MATXSCRIPT_AS_STR(kernel_##Prefix##_##OpName)) \
      .set_attr<TPrinterGlobalSymbol>("TPrinterGlobalSymbol", #Prefix "." #OpName)

}  // namespace builtin
}  // namespace ir
}  // namespace matxscript
