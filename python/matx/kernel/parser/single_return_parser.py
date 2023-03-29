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
from typing import Any, List, Dict, TYPE_CHECKING

from matx import ir as _ir
from matx.kernel.ops import OpRegistry
from matx.script import context as script_context
from .context import *
from .utils import build_span
from ..symbol.utils import compare as symbol_compare
from ...ir import AssignStmt
from ...ir.expr import *
from ...ir.tensor_stmt import ComputeBlock, BufferRegion

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class KernelSingleReturnParser(ast.NodeVisitor):
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

    def __init__(
            self,
            kernel_p: 'KernelParser',
            ndarray_context_table: Dict[str, NDArrayContext],
            shape_symbol_table: Dict[str, SymbolContext],
            return_ctx: NDArrayContext,
            node: script_context.ASTNode):
        self.kernel_p = kernel_p

        # necessary for reuse script functionality
        self.root_node = node
        self.context = None

        # for kernel use
        self.ndarray_context_table = ndarray_context_table
        self.shape_symbol_table = shape_symbol_table
        self.return_ctx = return_ctx

        self.var_stack = []
        self.ops = []
        self.read_ctx = []

    def generic_visit(self, node):
        """Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """
        raise NotImplementedError(f'This node is not supported now: {node}')

    def visit(self, node: ast.AST) -> Any:
        """Override method in ast.NodeVisitor"""
        method = "visit_" + node.__class__.__name__
        print(method)
        visitor = getattr(self, method, self.generic_visit)
        visit_res = visitor(node)
        return visit_res

    def parse_body(self, auto_add_return=False):
        body = []
        last_ast = None
        while len(self.context.node_stack[-1]) > 0:
            last_ast = self.context.node_stack[-1].pop()
            res = self.visit(last_ast)
            if res is not None:
                if not isinstance(res, _ir.Stmt):
                    raise SyntaxError('Every IR node here should be a stmt!')
                body.append(res)
            else:
                # ignore the stmt
                pass
        if (auto_add_return
                and (len(body) == 0 or not isinstance(last_ast, ast.Return))):
            body.append(self.visit(ast.Return(value=None)))
        return body

    @staticmethod
    def to_seq_stmt(body: List[_ir.Stmt], span: _ir.Span):
        if body is None or len(body) == 0:
            return _ir.SeqStmt([], span)
        return _ir.SeqStmt(body, span) if len(body) > 1 else body[0]

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.context = script_context.ScopeContext()
        self.context.new_scope(nodes=node.body)
        span = build_span(self.root_node, node)
        # add parameters of function
        for arg, ctx in self.ndarray_context_table.items():
            self.context.update_symbol(arg, ctx.script_ptr_var)
            self.context.func_params.append(ctx.script_ptr_var)

        # make dim variables as args
        for dim, dim_var in self.shape_symbol_table.items():
            self.context.update_symbol(dim, dim_var.script_var)
            self.context.func_params.append(dim_var.script_var)

        # append session_pointer_var
        pointer_var_name = "handle_2_71828182846"
        pointer_var = _ir.PrimVar(
            pointer_var_name,
            _ir.PrimType("handle")
        )
        self.context.update_symbol(pointer_var_name, pointer_var)
        self.context.func_params.append(pointer_var)

        body_stmts = self.parse_body(True)
        body_stmts = [body_stmts[0]]

        func = _ir.Function(
            self.context.func_params,
            # [HLOVar(x, ty=ObjectTypeNode), HLOVar(y, ty=ObjectTypeNode), handle_2_71828182846]
            [],  # [_ir.PrimCast("handle", _ir.const(0))],
            self.to_seq_stmt(body_stmts, span),
            # [CallNode(Op(ir.nd_module_add), [HLOVar(x, ty=ObjectTypeNode), HLOVar(y, ty=ObjectTypeNode)], []) -> NDArrayType]
            ret_type=None,
            span=span
        )
        func = func.with_attr(_ir.FuncAttr.kGlobalSymbol, node.name)
        self.context.pop_scope()
        return func

    def visit_Constant(self, node: ast.Constant) -> Any:
        if node.value is None:
            self.var_stack.append((None, None, None))
            return _ir.NoneExpr()
        elif isinstance(node.value, numbers.Number):
            dtype = get_dtype(node.value)

            const_scalar_ctx = ConstScalarContext(node.value,
                                                  PYTYPE_TO_KERNEL_TYPE[dtype],
                                                  build_span(self.root_node, node))

            self.var_stack.append(
                ("const", const_scalar_ctx, const_scalar_ctx.kernel_type))  # todo none for now
            return const_scalar_ctx.script_data_var
        else:
            raise NotImplementedError(f'Unsupported value {node.value}')

    # variables
    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        if name in self.shape_symbol_table:
            ctx = self.shape_symbol_table[name]
            self.var_stack.append((name, ctx, ctx.kernel_type))
            return ctx.script_var
        if name in self.ndarray_context_table:
            ctx = self.ndarray_context_table[name]
            self.var_stack.append((name, ctx, ctx.kernel_type))
            return ctx.script_ptr_var
        raise NotImplementedError(f'the type of {name} is not support')
        # return node.id

    # Expressions
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        raise NotImplementedError("visit_UnaryOp is not Implemented")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        # todo deal with other type
        # todo generate a intermediate dst to hold the data
        opname = type(node.op).__name__
        lhs_ir = self.visit(node.left)
        lhs, lhs_ctx, lhs_t = self.var_stack.pop()
        if not lhs_ctx.is_abstract_ctx():
            lhs_ir = lhs_ctx.script_data_var
        rhs_ir = self.visit(node.right)
        rhs, rhs_ctx, rhs_t = self.var_stack.pop()
        if not rhs_ctx.is_abstract_ctx():
            rhs_ir = rhs_ctx.script_data_var
        if is_ndarray_type(lhs_t) and is_ndarray_type(rhs_t):
            op_class = OpRegistry.get_bin_op(
                lhs_ctx.kernel_type, rhs_ctx.kernel_type, opname)
            op = op_class(lhs_ctx, rhs_ctx)
            dst_kernel_type = op.dst_kernel_type()
            var_info = (None, AbstractNDArrayContext(dst_kernel_type), dst_kernel_type)
            self.ops.append(op)
            self.var_stack.append(var_info)
            return op.ir_class(lhs_ir, rhs_ir)
        else:
            raise SyntaxError(f"bin op does not support {lhs_t} and {rhs_t}")
        # todo insert to ir

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        raise NotImplementedError("visit_BoolOp is not Implemented")

    def visit_Compare(self, node: ast.Compare) -> Any:
        raise NotImplementedError("visit_Compare is not Implemented")

    def visit_Call(self, node: ast.Call) -> Any:
        raise NotImplementedError("visit_Call is not Implemented")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        attr_name = node.attr
        if not isinstance(value, list):
            return [value, attr_name]
        return [*value, attr_name]

    def visit_Return(self, node: ast.Return) -> Any:
        # treat return as assign and
        # for now kernel does not return anything.
        if node.value is None:
            raise SyntaxError("return should not be empty")

        rt_ir = self.visit(node.value)

        # todo make compute block
        body = AssignStmt(self.return_ctx.script_data_var, rt_ir)
        result_shape = self._calculate_result_shape()
        if list(result_shape) != list(self.return_ctx.shape):
            raise RuntimeError(f"the marked shape {self.return_ctx.shape} "
                               f"is not equal to {result_shape}")
        return_range = self._make_range(result_shape)
        iter_vars_names = [f"__iter_{i}" for i in range(len(return_range))]
        iter_vars = [PrimIterVar(return_range[i], iter_vars_names[i])
                     for i in range(len(return_range))]
        reads = []  # todo update range
        # dst_shapes = [op.dst_kernel_type() for op in self.ops]
        for op in self.ops:
            if not op.lhs_ctx.is_abstract_ctx():
                buffer = op.lhs_ctx.buffer
                range_ = self._make_range(op.op.lhs_broad_cast_shape)
                reads.append(BufferRegion(buffer, range_))
            if not op.rhs_ctx.is_abstract_ctx():
                buffer = op.rhs_ctx.buffer
                range_ = self._make_range(op.op.rhs_broad_cast_shape)
                reads.append(BufferRegion(buffer, range_))

        writes = [BufferRegion(self.return_ctx.buffer, return_range)]

        return ComputeBlock(iter_vars, reads, writes, self.kernel_p.func_name, body)

    def _make_range(self, shape):
        rng = []
        for dim in shape:
            start = _ir.const(0)
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

    def _calculate_result_shape(self):
        longest_shape = []
        for op in self.ops:
            dst_shape = op.dst_shape()
            if len(dst_shape) > len(longest_shape):
                longest_shape = dst_shape
            elif len(dst_shape) == len(longest_shape) \
                    and symbol_compare(sum(dst_shape), sum(longest_shape)) > 0:
                longest_shape = dst_shape
        return longest_shape
