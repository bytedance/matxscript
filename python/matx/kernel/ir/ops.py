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
import ast

from matx.ir import generic as _generic
from .base import *
from .scalar import ConstScalarNode
from .utils import *
from ... import ir as _ir

_arithmetic_binop_maker = {
    ast.Add: lambda lhs, rhs, span: _generic.add(lhs, rhs, span),
    ast.Sub: lambda lhs, rhs, span: _generic.subtract(lhs, rhs, span),
    ast.Mult: lambda lhs, rhs, span: _generic.multiply(lhs, rhs, span),
    ast.Div: lambda lhs, rhs, span: _generic.divide(lhs, rhs, span),
    ast.FloorDiv: lambda lhs, rhs, span: _generic.floordiv(lhs, rhs, span),
    # ast.Mod: lambda lhs, rhs, span: _generic.floormod(lhs, rhs, span),
    # quick fix for mod sign issue
    ast.Mod: lambda lhs, rhs, span: _generic.floormod(lhs, rhs, span),
    ast.BitOr: lambda lhs, rhs, span: _generic.bitwise_or(lhs, rhs, span),
    ast.BitAnd: lambda lhs, rhs, span: _generic.bitwise_and(lhs, rhs, span),
    ast.BitXor: lambda lhs, rhs, span: _generic.bitwise_xor(lhs, rhs, span),
    # ast.LShift: lambda lhs, rhs, span: _generic.left_shift(lhs, rhs, span),
    # ast.RShift: lambda lhs, rhs, span: _generic.right_shift(lhs, rhs, span),
}

_unaryop_maker = {
    ast.USub: lambda operand, span: _generic.multiply(operand, _ir.const(-1), span),
    ast.Invert: lambda operand, span: _generic.bitwise_not(operand, span),
    ast.Not: lambda operand, span: _generic.op_not(operand, span)
}

_boolop_marker = {
    ast.Gt: lambda lhs, rhs, span: _generic.greater_than(lhs, rhs, span),
    ast.GtE: lambda lhs, rhs, span: _generic.greater_or_equal(lhs, rhs, span),
    ast.Lt: lambda lhs, rhs, span: _generic.less_than(lhs, rhs, span),
    ast.LtE: lambda lhs, rhs, span: _generic.less_or_equal(lhs, rhs, span),
    ast.Eq: lambda lhs, rhs, span: _generic.equal(lhs, rhs, span),
    ast.NotEq: lambda lhs, rhs, span: _generic.notequal(lhs, rhs, span),
    ast.Is: lambda lhs, rhs, span: _generic.op_is(lhs, rhs, span),
    ast.IsNot: lambda lhs, rhs, span: _generic.op_not(_generic.op_is(lhs, rhs, span), span),

    ast.And: lambda lhs, rhs, span: _generic.op_and(span, lhs, rhs),
    ast.Or: lambda lhs, rhs, span: _generic.op_or(span, lhs, rhs)
}


def reset_constant_dtype(lhs: ExpressionBaseNode, rhs: ExpressionBaseNode):
    def helper(const: ConstScalarNode, other: ExpressionBaseNode):
        const_kernel_t = const.kernel_type
        other_kernel_t = other.kernel_type
        const_dtype = const_kernel_t.dtype
        other_dtype = other_kernel_t.dtype
        if np.can_cast(
                const_dtype,
                other_dtype,
                "same_kind") and np.can_cast(
                const.value,
                other_dtype,
                "safe"):
            return ConstScalarNode(const.value, PYTYPE_TO_KERNEL_TYPE[other_dtype], const.span)
        else:
            raise SyntaxError(f"Cannot cast constant {const} from {const_dtype} to {other_dtype} "
                              f" which is the type of the other operand.")

    lhs_is_constant = isinstance(lhs, ConstScalarNode)
    rhs_is_constant = isinstance(rhs, ConstScalarNode)
    if lhs_is_constant == rhs_is_constant:
        return lhs, rhs
    if lhs_is_constant:
        return helper(lhs, rhs), rhs
    else:
        new_constant = helper(rhs, lhs)
        return lhs, new_constant


class CastOp(ExpressionBaseNode):

    def __init__(self, operand: ExpressionBaseNode, target_dtype, span):
        self.operand = operand
        self.target_dtype = target_dtype
        self.span = span
        self.operand_type = operand.kernel_type
        self.result_type = PYTYPE_TO_KERNEL_TYPE[target_dtype][self.operand_type.shape]
        super().__init__(self.result_type)

    def to_matx_ir(self, **kwargs):
        return _generic.cast(self.operand.to_matx_ir(**kwargs),
                             NPDTYPE_TO_STR[self.target_dtype], self.span)

    def buffer_regions(self, **kwargs):
        return self.operand.buffer_regions(**kwargs)


class BinaryOp(ExpressionBaseNode):

    def __init__(self, lhs: ExpressionBaseNode, rhs: ExpressionBaseNode, op_type, span):
        self.is_boolean_op = op_type in _boolop_marker
        if self.is_boolean_op:
            self.op = _boolop_marker[op_type]
        else:
            self.op = _arithmetic_binop_maker[op_type]
        lhs, rhs = reset_constant_dtype(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
        self.span = span
        self.lhs_type = lhs.kernel_type
        self.rhs_type = rhs.kernel_type
        self.lhs_dtype = get_dtype(self.lhs_type)
        self.rhs_dtype = get_dtype(self.rhs_type)
        self.lhs_shape = get_shape(self.lhs_type)
        self.rhs_shape = get_shape(self.rhs_type)
        if self.is_boolean_op:
            self.result_dtype = np.bool_
        elif op_type == ast.Div:
            self.result_dtype = (self.lhs_dtype(1) / self.rhs_dtype(1)).dtype.type
        else:
            self.result_dtype = np_result_dtype([self.lhs_dtype, self.rhs_dtype])
        result_shape, lhs_new_shape, rhs_new_shape = broadcast(self.lhs_shape, self.rhs_shape)
        self.result_shape = tuple(result_shape)
        self.result_type = NDArrayType(self.result_shape, self.result_dtype)
        self.lhs_broad_cast_shape = lhs_new_shape
        self.rhs_broad_cast_shape = rhs_new_shape
        super().__init__(self.result_type)
        self.shape = result_shape

    def to_matx_ir(self, **kwargs):
        matx_result = self.op(
            self.lhs.to_matx_ir(
                **kwargs),
            self.rhs.to_matx_ir(
                **kwargs),
            self.span)
        if matx_result.checked_type != _ir.PrimType(self.result_type.dtype_str()):
            matx_result = _generic.cast(matx_result, self.result_type.dtype_str(), self.span)
        return matx_result

    def buffer_regions(self, **kwargs):
        return self.lhs.buffer_regions(**kwargs) + self.rhs.buffer_regions(**kwargs)


class UnaryOp(ExpressionBaseNode):

    def __init__(self, operand: ExpressionBaseNode, op_type, span):
        self.op = _unaryop_maker[op_type]
        self.operand = operand
        self.span = span
        self.operand_type = operand.kernel_type
        self.operand_dtype = get_dtype(self.operand_type)
        self.result_dtype = np_result_dtype([self.operand_dtype])
        result_shape = self.operand.kernel_type.shape
        self.result_shape = tuple(result_shape)
        self.result_type = NDArrayType(self.result_shape, self.result_dtype)
        super().__init__(self.result_type)
        self.shape = result_shape

    def to_matx_ir(self, **kwargs):
        return self.op(self.operand.to_matx_ir(**kwargs), self.span)

    def buffer_regions(self, **kwargs):
        return self.operand.buffer_regions(**kwargs)
