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

from matx import ir as _ir
from matx.ir.tensor_stmt import decl_buffer
from matx.kernel.typing import *
from matx.kernel.typing import NDArrayType as kernelNDArrayT
from .base import *


class NDArrayNode(ExpressionBaseNode):
    def __init__(self, name: str, type_: kernelNDArrayT, shape_symbol_table: dict, span) -> None:
        assert is_ndarray_type(type_), 'syntax error'
        super().__init__(type_)
        self.name: str = name
        self.shape = type_.shape
        self.script_type = _ir.PointerType(_ir.PrimType(type_.dtype_str()))
        self.script_var = _ir.PrimVar(f"{name}", self.script_type, span)  # PTR_VAR
        buffer_shape = [dim if not is_symbol(dim) else shape_symbol_table[str(dim)].script_var
                        for dim in self.shape]
        self.idx = [_ir.const(0) for _ in range(len(self.shape))]
        self.buffer = decl_buffer(
            buffer_shape,
            dtype=type_.dtype_str(),
            name=name,
            data=self.script_var)
        self.range = []
        for dim in self.shape:
            start = _ir.const(0, "int64")
            if is_symbol(dim):
                symbol_ctx = shape_symbol_table[str(dim)]
                end = symbol_ctx.script_var
            elif dim is None:
                continue
            else:
                end = _ir.const(dim)
            rng_expr = _ir.RangeExpr(start, end)
            self.range.append(rng_expr)

    def to_matx_ir(self, iter_var=None, **kwargs):
        if iter_var is None:
            return self.script_var
        idx = iter_var[-len(self.shape):]
        return self.buffer.vload(tuple(idx))

    def buffer_regions(self, rng, **kwargs):
        return [_ir.BufferRegion(self.buffer, rng[-len(self.shape):])]

    def ndarrays(self):
        return [self]
