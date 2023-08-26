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

from typing import Any, TYPE_CHECKING

from matx import ir as _ir
from .general_ast_visitor import GeneralAstVisitor
from matx.ir.tensor_stmt import ComputeBlock

if TYPE_CHECKING:
    from matx.kernel.parser.function_parser import FunctionParser


class LoopAstVisitor(GeneralAstVisitor):
    allowed_ast_node = []

    def __init__(self,
                 func_parser: 'FunctionParser'):
        super().__init__(func_parser)
        self.loop_variable_map = {}
        self.writes = []
        self.reads = []
        # self.is_in_body = False

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        rt = super().visit_Subscript(node)
        """
        ctx = self.var_stack[-1]
        # if not self.is_in_body:
        #    return rt
        if not isinstance(ctx, AbstractNdarrayIndexingContext):
            return rt

        used_dims = len(ctx.idx)
        nd_ctx = self.ndarray_context_table[ctx.name]
        range_expr = self._make_range_with(nd_ctx.shape[-used_dims:])

        buffer_region = BufferRegion(ctx.buffer, range_expr)

        if isinstance(node.ctx, ast.Load):
            self.reads.append(buffer_region)
        elif isinstance(node.ctx, ast.Store):
            self.writes.append(buffer_region)
        else:
            raise SyntaxError(f"deleting {ctx.name} is not allowed")"""
        return rt

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
        range_args = self._get_for_loop_range(node.iter)
        start = ConstScalarNode(0, int64, span)
        end = range_args[0]
        step = ConstScalarNode(1, int64, span)
        if len(range_args) == 2:
            start = range_args[0]
            end = range_args[1]
        elif len(range_args) == 3:
            start = range_args[0]
            end = range_args[1]
            step = range_args[2]
        elif len(range_args) > 3:
            raise SyntaxError(
                f"the number of args for range should be at most 3, but get {len(range_args)}")

        # initialize loop variable
        iter_var_ctx = IterScalarNode(node.target.id, int64, start, end, step, span)
        self.loop_variable_map[iter_var_ctx.name] = (start, end, step)
        # the scalar is allocated in the table purely for reference during
        # the visiting of the loop body
        # (and for now we do not remove them from the table)
        # the ComputeBlock will actually allocate it and assign value to it.
        rt = self._allocate_scalar(iter_var_ctx.name, start, start.kernel_type, span)
        self.tmp_scalar_table[iter_var_ctx.name] = iter_var_ctx

        # visit body
        body_ir = self._visit_for_loop_body(node.body)

        if not is_top_loop:
            return body_ir
        compute_block = self._make_for_loop_compute_block(body_ir, span)
        self.loop_variable_map.clear()
        return compute_block

    def _make_for_loop_compute_block(self, rt_ir, span):
        # todo check shape
        loop_range = self._make_range()
        iter_vars_names = [self.tmp_scalar_table[name].script_var
                           for name in self.loop_variable_map.keys()]
        iter_vars = [PrimIterVar(loop_range[i], iter_vars_names[i]) for i in range(len(loop_range))]
        reads = [e for i in rt_ir for e in i.reads()]
        writes = [e for i in rt_ir for e in i.writes()]

        # for lhs_name, rhs in zip(self.loop_variable_map.keys(), iter_vars_names):
        #    lhs = self.tmp_scalar_table[lhs_name].script_var
        #    assign_iter_var.append(_ir.AssignStmt(lhs, rhs))

        body = _ir.SeqStmt([i.to_matx_ir() for i in rt_ir])

        return ComputeBlock(iter_vars, reads, writes, self.func_parser.func_name, body)

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
        else:
            rt_ir = []
            for node in body:
                rt_ir.append(self.visit(node))
        return rt_ir

    def _get_for_loop_range(self, rng):
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
            if not isinstance(a, (ast.Constant, ast.Name, ast.BinOp)):
                raise SyntaxError(
                    f"Args in range function should be constant, scalar, or symbol but get {type(a)}")
            a_ir = self.visit(a)
            if not isinstance(a_ir, (ScalarNode, SymbolNode, BinaryOp)):
                raise SyntaxError(
                    f"Args in range function should be constant, scalar, or symbol but get {type(a)}")
            args.append(a_ir)

        return args

    def _make_range(self):
        rng = []
        for key, r in self.loop_variable_map.items():
            rng.append(RangeExpr(r[0].to_matx_ir(), r[1].to_matx_ir(), r[2].to_matx_ir()))
        return rng
