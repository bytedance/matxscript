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

/*!
 * \file mlir_printer.h
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */
#pragma once
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "matxscript/ir/base.h"
#include "matxscript/ir/prim_expr.h"
#include "matxscript/ir/prim_ops.h"
#include "matxscript/ir/prim_var.h"
#include "matxscript/ir/printer/kernel_printer/mlir_printer.h"
#include "matxscript/ir/tensor_stmt.h"
#include "matxscript/ir/type.h"
#include "matxscript/runtime/dlpack.h"
#include "matxscript/runtime/object.h"

#include <matxscript/ir/expr_functor.h>
#include <matxscript/ir/function.h>
#include <matxscript/ir/stmt_functor.h>
#include <matxscript/ir/type_functor.h>
#include <matxscript/runtime/data_type.h>

namespace matxscript {
namespace ir {
namespace printer {
class MLIRTextPrinter;
class LinalgGenericPrinter {
  friend class MLIRTextPrinter;

 public:
  explicit LinalgGenericPrinter(MLIRTextPrinter* mlir_printer) : mlir_printer_(mlir_printer){};

 private:
  // default method

  void ComputeBlockToLinalgGeneric(const ComputeBlockNode* op, std::ostream& os);

  void PrintBufferArray(const Array<matxscript::ir::BufferRegion>& bufferArray,
                        const std::string& perfix_str,
                        std::ostream& os);
  void VisitRangeExpr_(const BufferRegion& buffer, const RangeExpr& rng, std::ostream& os);
  void GenAffineMap_(const Array<PrimIterVar>& iter_vars,
                     const Array<BufferRegion>& reads,
                     const Array<BufferRegion>& writes,
                     std::ostream& os);
  void VisitBufferRegionArray_(const Array<BufferRegion>& reads, std::ostream& os);
  void VisitComputBlockBody_(const Stmt& body, std::ostream& os);

  std::string GetPrimVarName(const BufferLoadNode* op);

  std::vector<const BufferRegionNode*> bufferRegionOrder;
  std::unordered_map<const BufferNode*, std::vector<const BufferRegionNode*>> regionMap;
  std::unordered_map<const BufferNode*, int> visitCounter;
  MLIRTextPrinter* mlir_printer_;
};
}  // namespace printer
}  // namespace ir
}  // namespace matxscript
