#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
from .ndarray import *
from ... import ir as _ir
from ...ir.expr import *
from typing import List


def make_range(shape, shape_symbol_table):
    rng = []
    for dim in shape:
        start = _ir.const(0, "int64")
        if is_symbol(dim):
            symbol_ctx = shape_symbol_table[str(dim)]
            end = symbol_ctx.script_var
        elif dim is None:
            continue
        else:
            end = _ir.const(dim)
        rng_expr = RangeExpr(start, end)
        rng.append(rng_expr)
    return rng


class AssignNDArrayNode(StatementBaseNode):

    def __init__(self, lhs: NDArrayNode, rhs: ExpressionBaseNode):
        if not isinstance(lhs, NDArrayNode):
            raise SyntaxError("some error")
        self.lhs: NDArrayNode = lhs
        self.rhs: ExpressionBaseNode = rhs
        self.range = self.lhs.range
        # todo check rhs lhs ranges are the same

        self.iter_var_names = [_ir.PrimVar(f"__iter_{i}", "int64")
                               for i in range(len(self.range))]
        self.iter_vars = [PrimIterVar(r, i) for r, i in zip(self.range, self.iter_var_names)]

    def to_matx_ir(self, **kwargs):
        if isinstance(self.lhs, NDArrayNode) and is_ndarray_type(self.rhs.kernel_type):
            return self.lhs.buffer.vstore(
                tuple(
                    self.iter_var_names), self.rhs.to_matx_ir(
                    iter_var=self.iter_var_names, **kwargs))
            # self.lhs
        # return self.lhs.

    def writes(self):
        return self.lhs.buffer_regions(rng=self.range)

    def reads(self):
        return self.rhs.buffer_regions(rng=self.range)

    def alocate_buffer(self):
        return []


class AssignScalarNode(StatementBaseNode):

    def __init__(self, lhs, rhs, span):
        self.lhs = lhs
        self.rhs = rhs
        self.span = span

    def to_matx_ir(self, **kwargs):
        if isinstance(self.lhs, NDArrayIndexingNode):
            return self.lhs.to_matx_ir(self.rhs)
        return _ir.AssignStmt(self.lhs.to_matx_ir(), self.rhs.to_matx_ir(), self.span)

    def writes(self):
        return self.lhs.buffer_regions()

    def reads(self):
        return self.rhs.buffer_regions()

    def alocate_buffer(self):
        return []


class ScalarAllocationNode(AssignScalarNode):

    def __init__(self, lhs, rhs, span):
        super().__init__(lhs, rhs, span)

    def to_matx_ir(self, **kwargs):
        return _ir.AllocaVarStmt(
            self.lhs.name,
            self.lhs.script_type,
            self.rhs.to_matx_ir(),
            self.span)

    def writes(self):
        return []

    def alocate_buffer(self):
        return []


class ScopedNDArrayAllocationNode(AssignNDArrayNode):
    def __init__(self, lhs: NDArrayNode, rhs: ExpressionBaseNode, body: List, span):
        super().__init__(lhs, rhs)
        self.span = span
        self.assign_stmt = AssignNDArrayNode(lhs, rhs)
        self.body = body

    def to_matx_ir(self, assign_stmt, **kwargs):
        return _ir.Allocate(
            self.lhs.buffer.data,
            self.lhs.kernel_type.dtype_str(),
            self.lhs.buffer_shape,
            const(
                1,
                dtype="uint1"),
            _ir.SeqStmt(
                [assign_stmt] +
                self.body,
                self.span))
