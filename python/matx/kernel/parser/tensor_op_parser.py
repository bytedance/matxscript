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
from typing import Any, Union, TYPE_CHECKING

import matx.kernel.graphIR as _gir
from matx.kernel.parser.general_parser import GeneralParser

if TYPE_CHECKING:
    from .function_visitor import FunctionVisitor


class TensorOpParser(GeneralParser):

    @staticmethod
    def can_parse(kernel_p: 'FunctionVisitor', node):
        if isinstance(node, (ast.Return, ast.AnnAssign, ast.Assign)):
            return True
        return False

    def __init__(self,
                 func_visitor: 'FunctionVisitor'):
        super().__init__(func_visitor)
        self.iter_vars_names = []

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = _gir.BinaryElementWiseOperator(type(node.op))
        lhs_ir = self.visit(node.left)
        rhs_ir = self.visit(node.right)
        result = op(lhs_ir, rhs_ir)[0]
        self.func_visitor.graph_nodes.append(result)
        self.func_visitor.graph_nodes.append(op)
        return result

    def visit_Assign(self, node: ast.Assign) -> Any:
        """
        stmt = super().visit_Assign(node)
        if isinstance(stmt, AssignScalarNode):
            return stmt.to_matx_ir()
        return self.make_compute_block(stmt)"""

        super().visit_Assign(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """
        stmt = super().visit_AnnAssign(node)
        if isinstance(stmt, ScalarAllocationNode):
            return stmt.to_matx_ir()
        if isinstance(stmt, ScopedNDArrayAllocationNode):
            cmptblk = self.make_compute_block(stmt.assign_stmt)
            return stmt.to_matx_ir(assign_stmt=cmptblk)"""
        super().visit_AnnAssign(node)

    def make_compute_block(self, stmt):
        """
        writes = stmt.writes()
        reads = stmt.reads()
        body = stmt.to_matx_ir()
        cmptblk = ComputeBlock(stmt.iter_vars, reads, writes, self.kernel_p.func_name, body)
        return cmptblk"""
        pass

    def visit_Return(self, node: ast.Return) -> Union[None, _gir.Node]:
        # treat return as assign and
        # for now kernel does not return anything.
        if node.value is None or _gir.utils.is_graph_ir_scalar(self.return_ctx):
            return super().visit_Return(node)

        if self.func_visitor.kernel_p.empty_return_signature:
            self.visit_return_without_signature(node)
        else:
            self.visit_return_with_signature(node)

    def visit_return_without_signature(self, node: ast.Return) -> Union[None, _gir.Node]:
        rt_ir = self.visit(node.value)
        dtype = rt_ir.dtype()
        shape = rt_ir.shape()
        self.func_visitor.kernel_p.return_types = _gir.utils.convert_to_kernel_type(rt_ir)
        self.func_visitor.make_return(shape, dtype)
        self.return_ctx = self.func_visitor.return_ctx
        if isinstance(self.return_ctx, _gir.Tensor):
            if self.return_ctx.name() != self.func_visitor.return_var_name:
                return None
            op = _gir.DeepCopyOperator()
            self.func_visitor.graph_nodes.append(op)
            op(self.return_ctx, rt_ir)
            return None
        else:
            raise RuntimeError(f"return {type(rt_ir)} is not support now")

    def visit_return_with_signature(self, node: ast.Return) -> Union[None, _gir.Node]:
        result_shape = self.func_visitor.return_types.shape
        if list(result_shape) != list(_gir.utils.unwrap_shape(self.return_ctx.shape())):
            raise RuntimeError(f"the marked shape {self.return_ctx.shape} "
                               f"is not equal to {result_shape}")

        rt_ir = self.visit(node.value)
        if list(result_shape) != list(_gir.utils.unwrap_shape(rt_ir.shape())):
            raise SyntaxError(
                f"The return shape is annotated as {result_shape} but get {rt_ir.shape()}")

        if isinstance(self.return_ctx, _gir.Tensor):
            if self.return_ctx.name() != self.func_visitor.return_var_name:
                return None
            op = _gir.DeepCopyOperator()
            self.func_visitor.graph_nodes.append(op)
            op(self.return_ctx, rt_ir)
            return None
        else:
            raise RuntimeError(f"return {type(rt_ir)} is not support now")
