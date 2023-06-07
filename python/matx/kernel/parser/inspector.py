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
from typing import Any, Dict, List, TYPE_CHECKING

from .for_loop_parser import ForLoopParser
from .single_return_parser import KernelSingleReturnParser
from .base_parser import BaseParser
from matx import ir as _ir
from .utils import *
from ..ir import *

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class BodyIterator:

    def __init__(self, node_stack, auto_add_return=False):
        self.body = []
        self.last_ast = None
        self.node_stack = node_stack
        self.auto_add_return = auto_add_return
        self.visited_added_return = False

    def has_next(self) -> bool:
        if not self.auto_add_return:
            return len(self.node_stack[-1]) > 0
        if self.visited_added_return:
            return False
        if len(self.node_stack[-1]) > 0:
            return True
        return self.auto_add_return and (
            len(self.body) == 0 or not isinstance(self.last_ast, ast.Return))

    def next(self):
        if len(self.node_stack[-1]) > 0:
            self.last_ast = self.node_stack[-1].pop()
            return self.last_ast
        self.visited_added_return = True
        return ast.Return(value=None)

    def push_ir(self, res):
        if res is not None:
            if isinstance(res, StatementBaseNode):
                res = res.to_matx_ir()
            if not isinstance(res, _ir.Stmt):
                raise SyntaxError('Every IR node here should be a stmt!')
            self.body.append(res)
        else:
            # ignore the stmt
            pass


class KernelInspector(ast.NodeVisitor):
    return_var_name = '__return_93502842947314__'

    def __init__(
            self,
            kernel_p: 'KernelParser',
            node: script_context.ASTNode):
        self.kernel_p = kernel_p

        # necessary for reuse script functionality
        self.root_node = node
        self.context = None

        # for kernel use
        self.arg_context_table: Dict[str, ExpressionBaseNode] = {}
        self.shape_symbol_table: Dict[str, SymbolNode] = {}
        self.tmp_scalar_table = {}
        self.tmp_ndarray_table = {}
        self.return_ctx = None
        self.return_types = kernel_p.return_types
        self.func_name = kernel_p.func_name

        # for checking
        self.visited_FunctionDef = False  # nested function definition is not allowed
        self.ast_nodes = []

        self.body_visiter = None

    def check_and_dispatch(self, node: ast.AST) -> Any:
        if isinstance(node, ast.For):
            p = ForLoopParser(self)
            return p.visit(node)
        elif KernelSingleReturnParser.can_parse(self, node):
            p = KernelSingleReturnParser(self)
            return p.visit(node)
        else:
            p = BaseParser(self)
            return p.visit(node)

    def declare_shape_var(self, type_annotation, span):
        if not isinstance(type_annotation, kernelNDArrayT):
            return
        shape_symbols = []
        for dim in type_annotation.shape:
            if not is_symbol(dim):
                continue
            if str(dim) in self.shape_symbol_table:
                continue
            sym_ctx = SymbolNode(dim, span)
            self.shape_symbol_table[str(dim)] = sym_ctx
            shape_symbols.append(sym_ctx)
        return shape_symbols

    def init_args(self, node: ast.FunctionDef) -> None:
        span = build_span(self.root_node, node)
        for arg, type_annotation in self.kernel_p.args.items():
            if is_scalar_type(type_annotation):
                scalar_ctx = ScalarNode(arg, type_annotation, span)
                self.arg_context_table[arg] = scalar_ctx
            elif is_ndarray_type(type_annotation):
                self.declare_shape_var(type_annotation, span)
                nd_ctx = NDArrayNode(arg, type_annotation, self.shape_symbol_table, span)
                self.arg_context_table[arg] = nd_ctx
            else:
                raise SyntaxError(f"right now only kernel ndarray are supported, "
                                  f"but get {type_annotation} for {arg}")

    def check_return(self, node: ast.FunctionDef) -> Any:
        # todo return scalar
        span = build_span(self.root_node, node)
        if self.kernel_p.return_types is None:
            raise SyntaxError("annotating return type is required for kernel functions")
        if not is_ndarray_type(self.kernel_p.return_types):
            raise SyntaxError("kernel function is supposed to return a kernel ndarray"
                              " or scalar(a.k.a) kernel ndarray with shape 1 "
                              f"but get {self.kernel_p.return_types}")
        for dim in self.kernel_p.return_types.shape:
            if is_symbol(dim) and str(dim) not in self.shape_symbol_table:
                raise SyntaxError(
                    f"{dim} in the return type is not defined in any of the shape in input args")
        if self.return_var_name in self.arg_context_table:
            self.return_ctx = self.arg_context_table[self.return_var_name]
            return

        if is_scalar_type(self.kernel_p.return_types):
            nd_ctx = ScalarNode(self.return_var_name, self.kernel_p.return_types, span)
            self.return_ctx = nd_ctx
        elif is_ndarray_type(self.kernel_p.return_types):
            nd_ctx = NDArrayNode(
                self.return_var_name,
                self.kernel_p.return_types,
                self.shape_symbol_table,
                span)
            self.arg_context_table[self.return_var_name] = nd_ctx
            self.return_ctx = nd_ctx

    @staticmethod
    def to_seq_stmt(body: List[_ir.Stmt], span: _ir.Span):
        if body is None or len(body) == 0:
            return _ir.SeqStmt([], span)
        return _ir.SeqStmt(body, span) if len(body) > 1 else body[0]

    def parse_body(self, auto_add_return=False):
        self.body_visiter = BodyIterator(self.context.node_stack, auto_add_return)
        while self.body_visiter.has_next():
            res = self.check_and_dispatch(self.body_visiter.next())
            self.body_visiter.push_ir(res)
        return self.body_visiter.body

    def continue_parse_as_scoped(self):
        ir_sofar = self.body_visiter.body
        self.body_visiter.body = []
        while self.body_visiter.has_next():
            res = self.check_and_dispatch(self.body_visiter.next())
            self.body_visiter.push_ir(res)
        scoped_ir = self.body_visiter.body
        self.body_visiter.body = ir_sofar
        return scoped_ir

    def visit_body(self, node: ast.FunctionDef):
        self.context = script_context.ScopeContext()
        self.context.new_scope(nodes=node.body)
        span_ = build_span(self.root_node, node)
        # add parameters of function
        nd_dim_map = {}
        for arg, ctx in self.arg_context_table.items():
            if not (isinstance(ctx, NDArrayNode) or isinstance(ctx, ScalarNode)):
                raise NotImplementedError("func parameters can only be markedas ndarray noe scalar")
            self.context.update_symbol(arg, ctx.script_var)
            self.context.func_params.append(ctx.script_var)
            if isinstance(ctx, NDArrayNode):
                nd_dim_map[ctx.script_var] = ctx.buffer

        # make dim variables as args
        for dim, dim_var in self.shape_symbol_table.items():
            self.context.update_symbol(dim, dim_var.script_var)
            self.context.func_params.append(dim_var.script_var)

        body_stmts = self.parse_body(True)

        func = _ir.PrimFunc(
            self.context.func_params,
            [],
            self.to_seq_stmt(body_stmts, span_),
            ret_type=None if not self.is_scalar_return() else self.return_ctx.script_type
        )
        func = func.with_attr(_ir.FuncAttr.kGlobalSymbol, node.name)
        func = func.with_attr(_ir.FuncAttr.kKernelFunctionParameterBinding, nd_dim_map)
        self.context.pop_scope()
        return func

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.visited_FunctionDef:
            raise SyntaxError("nested function def is not allowed.")
        self.visited_FunctionDef = True
        self.init_args(node)
        self.check_return(node)
        return self.visit_body(node)

    def is_scalar_return(self):
        return is_scalar_type(self.return_types)
