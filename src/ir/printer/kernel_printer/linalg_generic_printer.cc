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
 * \file linalg_printer.h
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */
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
#include "matxscript/ir/printer/kernel_printer/linalg_generic_printer.h"
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

using namespace ::matxscript::ir;
using namespace ::matxscript::runtime;

void LinalgGenericPrinter::VisitBufferRegionArray_(const Array<matxscript::ir::BufferRegion>& arr_,
                                                   std::ostream& os) {
  // region is ignored for now, and IMHO it should be ignored in this stage.
  std::stringstream types;
  for (int i = 0; i < arr_.size(); i++) {
    const auto& buffer = arr_[i]->buffer;
    const auto& region = arr_[i]->region;
    bufferRegionOrder.emplace_back(arr_[i].get());
    if (regionMap.find(buffer.get()) == regionMap.end()) {
      regionMap.emplace(buffer.get(), std::vector<const BufferRegionNode*>());
    }
    regionMap[buffer.get()].push_back(arr_[i].get());
    if (visitCounter.find(buffer.get()) == visitCounter.end()) {
      visitCounter.emplace(buffer.get(), 0);
    }
    mlir_printer_->PrintNodeName(buffer->data, os);
    types << mlir_printer_->ConvertTypeToMLIR(buffer);
    if (i != arr_.size() - 1) {
      os << ", ";
      types << ", ";
    }
  }
  if (arr_.size() > 0) {
    os << ": " << types.str();
  }
  return;
}

bool isInt(const matxscript::ir::PrimExpr& expr, const int expect) {
  if (expr->IsInstance<IntImmNode>()) {
    const auto& node = runtime::Downcast<IntImm>(expr);
    return node->value == expect;
  }
  return false;
}

void LinalgGenericPrinter::VisitRangeExpr_(const matxscript::ir::BufferRegion& buffer,
                                           const matxscript::ir::RangeExpr& rng,
                                           std::ostream& os) {
  const auto& start = rng->start;
  const auto& end = rng->stop;
  const auto& step = rng->step;
  // start has to be 0
  MXCHECK(isInt(start, 0)) << "The start (" << start << ") of range (" << rng << ") of buffer ("
                           << buffer << ") is not 0";
  // step has to be 1
  MXCHECK(isInt(step, 1)) << "The step (" << step << ") of range (" << rng << ") of buffer ("
                          << buffer << ") is not 1";
  // end
  if (end->IsInstance<PrimVarNode>()) {
    const auto& node = runtime::Downcast<PrimVar>(end);
    // todo check if it is iter var
    os << node->name_hint;
  } else {
    MXTHROW << "The end (" << end << ") of range (" << rng << ") of buffer (" << buffer
            << ") is not a iter var";
  }
}

void LinalgGenericPrinter::PrintBufferArray(const Array<matxscript::ir::BufferRegion>& bufferArray,
                                            const std::string& perfix_str,
                                            std::ostream& os) {
  for (int i = 0; i < bufferArray.size(); i++) {
    const auto& read_buffer = bufferArray[i];
    os << perfix_str;
    const auto& buffer = read_buffer->buffer;
    const auto& region = read_buffer->region;
    for (int i = 0; i < region.size(); i++) {
      const auto& range = region[i];
      VisitRangeExpr_(read_buffer, range, os);
      if (i != region.size() - 1) {
        os << ", ";
      }
    }
    os << ")>";
    if (i != bufferArray.size() - 1) {
      os << ", ";
    }
  }
}

void LinalgGenericPrinter::GenAffineMap_(const Array<matxscript::ir::PrimIterVar>& iter_vars,
                                         const Array<matxscript::ir::BufferRegion>& reads,
                                         const Array<matxscript::ir::BufferRegion>& writes,
                                         std::ostream& os) {
  os << "indexing_maps = [";

  // collect all iter vars and format them to affine_map<(i,j,k) -> (
  std::stringstream perfix;
  perfix << "affine_map<(";

  for (int i = 0; i < iter_vars.size(); i++) {
    const auto& start = iter_vars[i]->dom->start;
    const auto& stop = iter_vars[i]->dom->stop;
    const auto& step = iter_vars[i]->dom->step;
    // start has to be 0
    MXCHECK(isInt(start, 0)) << "The start (" << start << ") of iter_var (" << iter_vars[i]
                             << ") is not 0";
    // step has to be 1
    MXCHECK(isInt(step, 1)) << "The step (" << step << ") of iter_var (" << iter_vars[i]
                            << ")  is not 1";
    // stop has to be predefined symbol.
    if (!stop->IsInstance<PrimVarNode>()) {
      MXTHROW << "The end (" << stop << ") of iter_var (" << iter_vars[i]
              << ") is not a pre defined symbol";
    }
    perfix << stop;
    if (i != iter_vars.size() - 1) {
      perfix << ", ";
    }
  }
  perfix << ") -> (";
  auto perfix_str = perfix.str();

  // format for each input
  PrintBufferArray(reads, perfix_str, os);

  if (!writes.empty()) {
    os << ", ";
  }

  // format for each output
  PrintBufferArray(writes, perfix_str, os);

  os << "], iterator_types = [";
  // todo for now just assume they are parallel, deal with reduction later
  for (int i = 0; i < iter_vars.size(); i++) {
    os << "\"parallel\"";
    if (i != iter_vars.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
}

std::string LinalgGenericPrinter::GetPrimVarName(const BufferLoadNode* op) {
  const auto* bufferPtr = op->buffer.get();
  if (regionMap.find(bufferPtr) != regionMap.end() &&
      visitCounter.find(bufferPtr) != visitCounter.end()) {
    std::string element_name = bufferPtr->name.c_str() + std::to_string(visitCounter[bufferPtr]);
    visitCounter[bufferPtr]++;
    return "%_" + element_name;
  }
  MXTHROW << "[Linalg.generic] the corresponding buffer has not been recroded.";
  return "";
}

void LinalgGenericPrinter::VisitComputBlockBody_(const matxscript::ir::Stmt& body,
                                                 std::ostream& os) {
  os << "^bb0(";

  for (int i = 0; i < bufferRegionOrder.size(); i++) {
    const auto bufferRegionPtr = bufferRegionOrder.at(i);
    // bufferregion maybe over the same buffer.

    const auto& buffer = bufferRegionPtr->buffer;
    std::string element_name = buffer->name.c_str();
    const auto& regionArray = regionMap[buffer.get()];
    const int idx =
        std::find(regionArray.begin(), regionArray.end(), bufferRegionPtr) - regionArray.begin();
    element_name = "%_" + element_name + std::to_string(idx);
    mlir_printer_->insert_or_assign_expr_name_map_(bufferRegionPtr, element_name);
    os << element_name << ": " << mlir_printer_->ConvertTypeToMLIR(buffer->dtype);
    if (i != bufferRegionOrder.size() - 1) {
      os << ", ";
    }
  }
  // "%a: f32, %b: f32, %c: f32"
  os << "):" << std::endl;
  mlir_printer_->VisitStmt(body, os);
}

void LinalgGenericPrinter::ComputeBlockToLinalgGeneric(const ComputeBlockNode* op,
                                                       std::ostream& os) {
  /**
   *   Array<PrimIterVar> iter_vars;
   *   Array<BufferRegion> reads;
   *   Array<BufferRegion> writes;
   *   StringRef name_hint;
   *   Stmt body;
   */
  if (op->reads.empty()) {
    MXTHROW << "Not able to convert to linalg.generic. The reads in the compute block is empty";
    return;
  }

  if (op->writes.empty()) {
    MXTHROW << "Not able to convert to linalg.generic. The writes in the compute block is empty";
    return;
  }

  mlir_printer_->NewScope();
  os << "linalg.generic {";
  // visit iter_var (affine_map&iterator_types)
  GenAffineMap_(op->iter_vars, op->reads, op->writes, os);
  os << "}" << std::endl;
  // visit inputs
  os << "                    ins(";
  VisitBufferRegionArray_(op->reads, os);
  os << ')' << std::endl;
  // visit outputs
  os << "                    outs(";
  VisitBufferRegionArray_(op->writes, os);
  os << ')' << std::endl;
  os << "{" << std::endl;
  // visit computblock
  VisitComputBlockBody_(op->body, os);
  os << "}" << std::endl;
  mlir_printer_->PopScope();
  bufferRegionOrder.clear();
}

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
