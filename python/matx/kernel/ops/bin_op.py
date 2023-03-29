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

from functools import partial

from .base_op import *
from .registry.registry import OpRegistry
from .utils import *
from ... import ir as _ir
from ...ir.expr import *


class ArithmeticBinaryOp(KernelBaseOp):

    def __init__(self, lhs_type, rhs_type):
        super().__init__()
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type
        self.lhs_dtype = get_dtype(lhs_type)
        self.rhs_dtype = get_dtype(rhs_type)
        self.lhs_shape = get_shape(lhs_type)
        self.rhs_shape = get_shape(rhs_type)
        self.result_dtype = self._result_dtype()
        result_shape, lhs_new_shape, rhs_new_shape = broadcast(self.lhs_shape, self.rhs_shape)
        self.result_shape = result_shape
        self.result_type = NDArrayType(self.result_shape, self.result_dtype)
        self.lhs_broad_cast_shape = lhs_new_shape
        self.rhs_broad_cast_shape = rhs_new_shape

    def _result_dtype(self):
        result_dtype = np_result_dtype([self.lhs_dtype, self.rhs_dtype])
        return result_dtype


class AddOp(ArithmeticBinaryOp):
    opname = 'Add'
    operator = '+'
    ir_class = PrimAdd


class SubOp(ArithmeticBinaryOp):
    opname = 'Sub'
    operator = '-'
    ir_class = PrimSub


class MultOp(ArithmeticBinaryOp):
    opname = 'Mult'
    operator = '*'
    ir_class = PrimMul


class DivOp(ArithmeticBinaryOp):
    opname = 'Div'
    operator = '/'
    ir_class = PrimDiv


class ArrayArrayBinaryOp:
    def __init__(self, lhs_ctx, rhs_ctx, op_class: ArithmeticBinaryOp.__class__):
        self.lhs_ctx = lhs_ctx
        self.rhs_ctx = rhs_ctx
        self.lhs_kernel_type = lhs_ctx.kernel_type
        self.rhs_kernel_type = rhs_ctx.kernel_type
        self.op: ArithmeticBinaryOp = op_class(self.lhs_kernel_type, self.rhs_kernel_type)
        self.ir_class = self.op.ir_class
        self.opname = self.op.operator

    def result_type(self):
        return self.op.result_type

    def lhs_range(self):
        return [RangeExpr(_ir.const(0), PrimVar(str(dim), "int64"))
                for dim in self.op.lhs_broad_cast_shape]

    def rhs_range(self):
        return [RangeExpr(_ir.const(0), PrimVar(str(dim), "int64"))
                for dim in self.op.rhs_broad_cast_shape]

    def dst_range(self):
        return [RangeExpr(_ir.const(0), PrimVar(str(dim), "int64")) for dim in self.op.result_shape]

    def dst_kernel_type(self):
        return self.op.result_type

    def dst_shape(self):
        return self.op.result_shape

    # def __call__(self, dst_context):
    #    lhs_ranges = [RangeExpr(_ir.const(0), PrimVar(str(dim), "int64"))
    #                  for dim in self.op.lhs_broad_cast_shape]
    #    rhs_ranges = [RangeExpr(_ir.const(0), PrimVar(str(dim), "int64"))
    #                  for dim in self.op.rhs_broad_cast_shape]
    #    dst_ranges = [RangeExpr(_ir.const(0), PrimVar(str(dim), "int64"))
    #                  for dim in self.op.result_shape]
    #    iter_vars = [PrimIterVar(dom, None) for dom in dst_ranges]
    #    lhs_buffer_region = BufferRegion(self.lhs_context.buffer, lhs_ranges)
    #    rhs_buffer_region = BufferRegion(self.rhs_context.buffer, rhs_ranges)
    #    dst_buffer_region = BufferRegion(dst_context.buffer, dst_ranges)
    #    reads = [lhs_buffer_region, rhs_buffer_region]
    #    writes = [dst_buffer_region]
    #    name_hint = f"{self.lhs_context.name} {self.op.opname} {self.rhs_context.name}"
    #    element_op = self.op.ir_class(
    #        self.lhs_context.script_data_var,
    #        self.rhs_context.script_data_var)
    #    body = AssignStmt(dst_context.script_data_var, element_op)
    #    compute_block = ComputeBlock(iter_vars, reads, writes, name_hint, body)
    #    return lhs_buffer_region, rhs_buffer_region


def make_bin_op(op_class):
    def op(func):
        return partial(func, op_class=op_class)

    OpRegistry.add_bin_op(
        op(ArrayArrayBinaryOp),
        'NDArrayType',
        'NDArrayType',
        op_class.opname)
