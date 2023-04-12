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

from typing import Any, Dict, TYPE_CHECKING

import matx.ir as _ir
import matx.kernel.ir as kernel_ir
from matx.script import context as script_context
from .base_parser import BaseParser
from ..ir import *
from ...ir.tensor_stmt import ComputeBlock

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class KernelSingleReturnParser(BaseParser):
    allowed_ast_node = [
        ast.Return,
        ast.BinOp,
        ast.Add,
        ast.Div,
        ast.Sub,
        ast.Mult,
        ast.UnaryOp,
        ast.BinOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant]

    def __init__(self,
                 kernel_p: 'KernelParser',
                 ndarray_context_table: Dict[str,
                 ExpressionBaseNode],
                 shape_symbol_table: Dict[str,
                 SymbolNode],
                 return_ctx: ExpressionBaseNode,
                 node: script_context.ASTNode):
        super().__init__(kernel_p, ndarray_context_table, shape_symbol_table, return_ctx, node)
        self.iter_vars_names = []

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        # todo deal with other type
        # todo generate a intermediate dst to hold the data
        opname = type(node.op).__name__
        lhs_ir = self.visit(node.left)
        rhs_ir = self.visit(node.right)
        op = kernel_ir.ArithmeticBinaryOp(lhs_ir, rhs_ir, type(node.op), self.build_span(node))
        # todo update iter var names
        # todo update BufferRegion to reads

        """
                op_class = OpRegistry.get_bin_op(lhs_ctx.kernel_type, rhs_ctx.kernel_type, opname)
                op = op_class(lhs_ctx, rhs_ctx)
                dst_kernel_type = op.dst_kernel_type()
                result_dtype = op.op.result_type.dtype_str()
                var_info = AbstractNDArrayContext(dst_kernel_type)
                self.var_stack.append(var_info)
                if not lhs_ctx.is_abstract_ctx():
                    lhs_ir = lhs_ctx.read_at(self.iter_vars_names[-len(lhs_ctx.shape):])
                    range_ = self._make_range(op.op.lhs_broad_cast_shape)
                    self.reads.append(BufferRegion(lhs_ctx.buffer, range_))
                if not rhs_ctx.is_abstract_ctx():
                    rhs_ir = rhs_ctx.read_at(self.iter_vars_names[-len(rhs_ctx.shape):])
                    range_ = self._make_range(op.op.rhs_broad_cast_shape)
                    self.reads.append(BufferRegion(rhs_ctx.buffer, range_))
                # return op.ir_class(lhs_ir, rhs_ir)
                return self._binop_maker[type(node.op)](lhs_ir, rhs_ir, self.build_span(node))
                # todo insert to ir
        """
        return op

    def visit_Return(self, node: ast.Return) -> Any:
        # treat return as assign and
        # for now kernel does not return anything.
        if node.value is None:
            raise SyntaxError("return should not be empty")

        # todo make compute block
        result_shape = self.kernel_p.return_types.shape
        if list(result_shape) != list(self.return_ctx.shape):
            raise RuntimeError(f"the marked shape {self.return_ctx.shape} "
                               f"is not equal to {result_shape}")
        """
        return_range = self._make_range(result_shape)
        self.iter_vars_names = [
            _ir.PrimVar(
                f"__iter_{i}",
                "int64") for i in range(
                len(return_range))]
        iter_vars = [PrimIterVar(return_range[i], self.iter_vars_names[i])
                     for i in range(len(return_range))]
"""
        rt_ir = self.visit(node.value)
        if list(result_shape) != list(rt_ir.shape):
            raise SyntaxError(
                f"The return shape is annotated as {result_shape} but get {rt_ir.shape}")

        assign_op = kernel_ir.AssignNDArrayNode(self.return_ctx, rt_ir)

        writes = assign_op.writes()
        reads = assign_op.reads()

        # if isinstance(node.value, ast.Name):
        #    rt_ir = rt_ctx.read_at(self.iter_vars_names)
        # if rt_ctx.kernel_type.dtype != self.return_ctx.kernel_type.dtype:
        #    rt_ir = _ir.PrimCast(self.return_ctx.kernel_type.dtype_str(), rt_ir)
        # body = self.return_ctx.write_at(self.iter_vars_names, rt_ir)
        body = assign_op.to_matx_ir()

        return ComputeBlock(assign_op.iter_vars, reads, writes, self.kernel_p.func_name, body)

    def _make_range(self, shape):
        rng = []
        for dim in shape:
            start = _ir.const(0, "int64")
            if is_symbol(dim):
                symbol_ctx = self.shape_symbol_table[str(dim)]
                end = symbol_ctx.script_var
            elif dim is None:
                continue
            else:
                end = _ir.const(dim)
            rng_expr = RangeExpr(start, end)
            rng.append(rng_expr)
        return rng
