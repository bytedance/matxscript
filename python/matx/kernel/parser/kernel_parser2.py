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
from __future__ import annotations

from ast import *
from typing import Any, Dict, TYPE_CHECKING

from matx import ir as _ir
from matx.script import context as script_context
from .base_parser import BaseParser
from .context import *
from ..typing import *
from ...ir import AssignStmt
from ...ir.expr import *
from ...ir.tensor_stmt import ComputeBlock, BufferRegion

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class KernelParser2(BaseParser):

    def __init__(self, kernel_p: 'KernelParser', ndarray_context_table: Dict[str, NDArrayContext],
                 shape_symbol_table: Dict[str, SymbolContext], return_ctx: NDArrayContext,
                 node: script_context.ASTNode):
        super().__init__(kernel_p, ndarray_context_table, shape_symbol_table, return_ctx, node)

    def visit_Return(self, node: Return) -> Any:
        if node.value is None:
            raise SyntaxError("return should not be empty")

        rt_ir = self.visit(node.value)
        rt_ctx = self.var_stack.pop()

        # todo make compute block
        body = AssignStmt(self.return_ctx.data.script_var, rt_ir)
        result_shape = rt_ctx.shape
        if list(result_shape) != list(self.return_ctx.shape):
            raise RuntimeError(f"the marked shape {self.return_ctx.shape} "
                               f"is not equal to {result_shape}")
        return_range = self._make_range(result_shape)
        iter_vars_names = [f"__iter_{i}" for i in range(len(return_range))]
        iter_vars = [PrimIterVar(return_range[i], iter_vars_names[i])
                     for i in range(len(return_range))]

        writes = [BufferRegion(self.return_ctx.buffer, return_range)]

        return ComputeBlock(iter_vars, self.reads, writes, self.kernel_p.func_name, body)

    def visit_For(self, node: For) -> Any:
        target = node.target
        rng = node.iter
        body = node.body
        orelse = node.orelse
        if not isinstance(target, Name):
            raise SyntaxError(f"The iteration variable should not be {type(target)}. "
                              f"It is supposed to be a single variable")
        if len(orelse) != 0:
            raise SyntaxError(f"The else part in for loop is not allowed")

        range_args = self._check_range(rng)
        start = (_ir.const(0, "int64"), ConstScalarContext(0, int64, self.build_span(node)))
        end = range_args[0]
        step = (_ir.const(1, "int64"), ConstScalarContext(1, int64, self.build_span(node)))
        if len(range_args) == 2:
            start = range_args[0]
            end = range_args[1]
        elif len(range_args) == 3:
            start = range_args[0]
            end = range_args[1]
            step = range_args[2]
        else:
            raise SyntaxError(
                f"the number of args for range should be at most 3, but get {len(range_args)}")

    def _check_range(self, rng):
        if not isinstance(rng, Call):
            raise SyntaxError(
                f"The generator or iterator is only allowed to be python builtin range, "
                f"but get {type(rng)}")
        if not isinstance(rng.func, Name):
            raise SyntaxError(
                f"The generator or iterator is only allowed to be python builtin range, "
                f"but get {type(rng.func)}")
        if rng.func.id != 'range':
            raise SyntaxError(
                f"The generator or iterator is only allowed to be python builtin range, "
                f"but get {rng.func.id}")

        args = []
        for a in rng.args:
            # todo support symbolic expression
            if not isinstance(a, Constant):
                raise SyntaxError(f"Args in range function should be constant but get {type(a)}")
            a_ir = self.visit(a)
            a_ctx = self.var_stack.pop()
            args.append((a_ir, a_ctx))

        return args
