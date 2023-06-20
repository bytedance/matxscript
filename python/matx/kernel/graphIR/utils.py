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

_arithmetic_binop_maker = {
    ast.Add: lambda lhs, rhs, span: _generic.add(lhs, rhs, span),
    ast.Sub: lambda lhs, rhs, span: _generic.subtract(lhs, rhs, span),
    ast.Mult: lambda lhs, rhs, span: _generic.multiply(lhs, rhs, span),
    ast.Div: lambda lhs, rhs, span: _generic.divide(lhs, rhs, span),
    ast.FloorDiv: lambda lhs, rhs, span: _generic.floordiv(lhs, rhs, span),
    # ast.Mod: lambda lhs, rhs, span: _generic.floormod(lhs, rhs, span),
    # quick fix for mod sign issue
    ast.Mod: lambda lhs, rhs, span: _generic.floormod(_generic.add(_generic.floormod(lhs, rhs, span), rhs, span), rhs, span),
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


def is_scalar():
    pass
