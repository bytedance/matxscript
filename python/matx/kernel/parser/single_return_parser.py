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

from matx.script import context as script_context
from .base_parser import ParserBase
from .context import *
from ...ir import AssignStmt
from ...ir.expr import *
from ...ir.tensor_stmt import ComputeBlock, BufferRegion

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class KernelSingleReturnParser(ParserBase):
    allowed_ast_node = [
        ast.Return,
        ast.BinOp,
        ast.Add,
        ast.Div,
        ast.UnaryOp,
        ast.BinOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant]

    def __init__(self, kernel_p: 'KernelParser', ndarray_context_table: Dict[str, NDArrayContext],
                 shape_symbol_table: Dict[str, SymbolContext], return_ctx: NDArrayContext,
                 node: script_context.ASTNode):
        super().__init__(kernel_p, ndarray_context_table, shape_symbol_table, return_ctx, node)

    def visit_Return(self, node: ast.Return) -> Any:
        # treat return as assign and
        # for now kernel does not return anything.
        if node.value is None:
            raise SyntaxError("return should not be empty")

        rt_ir = self.visit(node.value)
        rt_ctx = self.var_stack.pop()

        # todo make compute block
        body = AssignStmt(self.return_ctx.script_data_var, rt_ir)
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
