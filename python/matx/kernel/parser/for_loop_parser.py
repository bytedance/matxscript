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

import ast
from typing import Any, Dict, TYPE_CHECKING

from matx import ir as _ir
from matx.script import context as script_context
from .base_parser import BaseParser
from .context import *
from ...ir.expr import *
from ...ir.tensor_stmt import ComputeBlock

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class ForLoopParser(BaseParser):
    allowed_ast_node = []

    def __init__(self, kernel_p: 'KernelParser', ndarray_context_table: Dict[str, NDArrayContext],
                 shape_symbol_table: Dict[str, SymbolContext], return_ctx: NDArrayContext,
                 node: script_context.ASTNode):
        super().__init__(kernel_p, ndarray_context_table, shape_symbol_table, return_ctx, node)
        self.loop_variable_map = {}
        self.loop_statements = []

    def visit_For(self, node: ast.For) -> Any:
        # top loop built the computeblock
        span = self.build_span(node)
        is_top_loop = len(self.loop_variable_map) == 0

        # has to be only one iter variable
        if not isinstance(node.target, ast.Name):
            raise SyntaxError(f"The iteration variable should not be {type(node.target)}. "
                              f"It is supposed to be a single variable")
        # no orelse part for for-loop
        if len(node.orelse) != 0:
            raise SyntaxError(f"The else part in for loop is not allowed")

        # get the start, end, and step
        range_args = self._check_range(node.iter)
        start = ConstScalarContext(0, int64, span)
        end = range_args[0][1]
        step = ConstScalarContext(1, int64, span)
        if len(range_args) == 2:
            start = range_args[0][1]
            end = range_args[1][1]
        elif len(range_args) == 3:
            start = range_args[0][1]
            end = range_args[1][1]
            step = range_args[2][1]
        elif len(range_args) > 3:
            raise SyntaxError(
                f"the number of args for range should be at most 3, but get {len(range_args)}")

        iter_var_ctx = ScalarContext(node.target.id, int64, span)
        self.loop_variable_map[iter_var_ctx.name] = (start, end, step)
        # the scalar is allocated in the table purely for reference during
        # the visiting of the loop body
        # (and for now we do not remove them from the table)
        # the ComputeBlock will actually allocate it and assign value to it.
        rt = self._allocate_scalar(iter_var_ctx.name, start.script_var,
                                   start, start.kernel_type, span)
        # self.loop_statements.append(rt)

        rt_ir, rt_ctx = self._visit_for_loop_body(node.body)
        if not is_top_loop:
            self.var_stack.append(rt_ctx)
            return rt_ir
        compute_block = self._make_for_loop_compute_block(rt_ir, rt_ctx, span)
        self.loop_variable_map.clear()
        return compute_block

    def _make_for_loop_compute_block(self, rt_ir, rt_ctx, span):
        # todo check shape
        loop_range = self._make_range()
        iter_vars_names = [self.tmp_scalar_table[name].script_var
                           for name in self.loop_variable_map.keys()]
        iter_vars = [PrimIterVar(loop_range[i], iter_vars_names[i]) for i in range(len(loop_range))]
        reads = []
        writes = []
        assign_iter_var = []
        # for lhs_name, rhs in zip(self.loop_variable_map.keys(), iter_vars_names):
        #    lhs = self.tmp_scalar_table[lhs_name].script_var
        #    assign_iter_var.append(_ir.AssignStmt(lhs, rhs))

        body = _ir.SeqStmt(self.loop_statements + assign_iter_var + rt_ir)

        return ComputeBlock(iter_vars, reads, writes, self.kernel_p.func_name, body)

    def _visit_for_loop_body(self, body):
        nested_for = None
        others = 0
        for node in body:
            if isinstance(node, ast.For):
                if nested_for is not None:
                    raise SyntaxError(
                        "In the body if for loop, only one nested for loop is allowed")
                nested_for = node
            elif isinstance(node, (ast.Assign, ast.If, ast.Pass, ast.AnnAssign)):
                others += 1
            else:
                raise SyntaxError(
                    f"In the body if for loop, "
                    f"either another for loop or assignment is allowed but get {type(node)}")
        if nested_for is not None and others != 0:
            raise SyntaxError(f"In the body if for loop, "
                              f"either another for loop or assignment is allowed but not both")

        if nested_for is not None:
            rt_ir = self.visit(nested_for)
            rt_ctx = self.var_stack.pop()
        else:
            rt_ir = []
            rt_ctx = []
            for node in body:
                rt_ir.append(self.visit(node))
                rt_ctx.append(self.var_stack.pop())
        return rt_ir, rt_ctx

    def _check_range(self, rng):
        # the range can only be the builtin range
        if not isinstance(rng, ast.Call):
            raise SyntaxError(
                f"The generator or iterator is only allowed to be python builtin range, "
                f"but get {type(rng)}")
        if not isinstance(rng.func, ast.Name):
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
            if not isinstance(a, (ast.Constant, ast.Name)):
                raise SyntaxError(
                    f"Args in range function should be constant, scalar, or symbol but get {type(a)}")
            a_ir = self.visit(a)
            a_ctx = self.var_stack.pop()
            if not isinstance(a_ctx, (ScalarContext, SymbolContext)):
                raise SyntaxError(
                    f"Args in range function should be constant, scalar, or symbol but get {type(a)}")
            args.append((a_ir, a_ctx))

        return args

    def _make_range(self):
        rng = []
        for key, r in self.loop_variable_map.items():
            rng.append(RangeExpr(r[0].script_var, r[1].script_var, r[2].script_var))
        return rng
