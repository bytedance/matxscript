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
from typing import Any, TYPE_CHECKING

import matx.ir as _ir
import matx.kernel.ir as kernel_ir
from .base_parser import BaseParser

from ..ir import *
from ...ir.tensor_stmt import ComputeBlock

if TYPE_CHECKING:
    from .inspector import KernelInspector


class KernelSingleReturnParser(BaseParser):

    @staticmethod
    def can_parse(kernel_p: 'KernelInspector', node):
        if isinstance(node, (ast.Return, ast.AnnAssign, ast.Assign)):
            return True
        return False

    def __init__(self,
                 kernel_p: 'KernelInspector'):
        super().__init__(kernel_p)
        self.iter_vars_names = []

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs_ir = self.visit(node.left)
        rhs_ir = self.visit(node.right)
        op = kernel_ir.BinaryOp(lhs_ir, rhs_ir, type(node.op), self.build_span(node))
        return op

    def visit_Assign(self, node: ast.Assign) -> Any:
        stmt = super().visit_Assign(node)
        if isinstance(stmt, AssignScalarNode):
            return stmt.to_matx_ir()
        return self.make_compute_block(stmt)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        stmt = super().visit_AnnAssign(node)
        if isinstance(stmt, ScalarAllocationNode):
            return stmt.to_matx_ir()
        if isinstance(stmt, ScopedNDArrayAllocationNode):
            cmptblk = self.make_compute_block(stmt.assign_stmt)
            return stmt.to_matx_ir(assign_stmt=cmptblk)
        raise NotImplementedError("not support assassign")

    def make_compute_block(self, stmt):
        writes = stmt.writes()
        reads = stmt.reads()
        body = stmt.to_matx_ir()
        cmptblk = ComputeBlock(stmt.iter_vars, reads, writes, self.kernel_p.func_name, body)
        return cmptblk

    def visit_Return(self, node: ast.Return) -> Any:
        # treat return as assign and
        # for now kernel does not return anything.
        if node.value is None or is_scalar_type(self.return_ctx.kernel_type):
            return super().visit_Return(node)

        result_shape = self.kernel_p.return_types.shape
        if list(result_shape) != list(self.return_ctx.shape):
            raise RuntimeError(f"the marked shape {self.return_ctx.shape} "
                               f"is not equal to {result_shape}")

        rt_ir = self.visit(node.value)
        if list(result_shape) != list(rt_ir.shape):
            raise SyntaxError(
                f"The return shape is annotated as {result_shape} but get {rt_ir.shape}")

        if isinstance(self.return_ctx, NDArrayNode):
            if self.return_ctx.name != self.kernel_p.return_var_name:
                return _ir.ReturnStmt(NoneExpr())
            stmt = AssignNDArrayNode(self.return_ctx, rt_ir)
            cmptblk = self.make_compute_block(stmt)
            return _ir.SeqStmt([cmptblk, _ir.ReturnStmt(NoneExpr())])
        else:
            raise RuntimeError(f"return {type(rt_ir)} is not support now")
