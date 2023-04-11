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
from typing import Any, Dict, TYPE_CHECKING

from .context import *
from .for_loop_parser import ForLoopParser
from .single_return_parser import KernelSingleReturnParser
from .utils import *
from ..ir import NDArrayNode, ScalarNode

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class KernelInspector(ast.NodeVisitor):
    return_var_name = '__return_93502842947314__'

    def __init__(
            self,
            kernel_p: 'KernelParser',
            node: script_context.ASTNode):
        self.kernel_p = kernel_p

        # necessary for reuse script functionality
        self.root_node = node

        self.kernel_parser = None

        # for kernel use
        self.ndarray_context_table: Dict[str, NDArrayContext] = {}
        self.shape_symbol_table: Dict[str, SymbolContext] = {}
        self.return_ctx = None

        # for checking
        self.visited_FunctionDef = False  # nested function definition is not allowed
        self.ast_nodes = []

    def check_and_dispatch(self, node: ast.AST) -> Any:
        self.visit(node)
        return self.kernel_parser(
            self.kernel_p,
            self.ndarray_context_table,
            self.shape_symbol_table,
            self.return_ctx,
            self.root_node)

    def generic_visit(self, node):
        self.ast_nodes.append(node.__class__)
        return super().generic_visit(node)

    def visit(self, node: ast.AST) -> Any:
        """Override method in ast.NodeVisitor"""
        method = "visit_" + node.__class__.__name__
        print(method)
        visitor = getattr(self, method, self.generic_visit)
        visit_res = visitor(node)
        return visit_res

    def declare_shape_var(self, type_annotation, span):
        if not isinstance(type_annotation, kernelNDArrayT):
            return
        shape_symbols = []
        for dim in type_annotation.shape:
            if not is_symbol(dim):
                continue
            if str(dim) in self.shape_symbol_table:
                continue
            sym_ctx = SymbolContext(dim, span)
            self.shape_symbol_table[str(dim)] = sym_ctx
            shape_symbols.append(sym_ctx)
        return shape_symbols

    def init_args(self, node: ast.FunctionDef) -> None:
        span = build_span(self.root_node, node)
        for arg, type_annotation in self.kernel_p.args.items():
            if is_scalar_type(type_annotation):
                scalar_ctx = ScalarNode(arg, type_annotation, span)
                self.ndarray_context_table[arg] = scalar_ctx
            elif is_ndarray_type(type_annotation):
                self.declare_shape_var(type_annotation, span)
                nd_ctx = NDArrayNode(arg, type_annotation, self.shape_symbol_table, span)
                self.ndarray_context_table[arg] = nd_ctx
            else:
                raise SyntaxError(f"right now only kernel ndarray are supported, "
                                  f"but get {type_annotation} for {arg}")

    def check_body(self, node: ast.FunctionDef) -> None:
        stmts = node.body
        # case 1 one line of return
        if len(stmts) == 1 and isinstance(stmts[0], ast.Return):
            self.kernel_parser = KernelSingleReturnParser
            self.visit(stmts[0])
            print(self.ast_nodes)
            for node in self.ast_nodes:
                if node not in KernelSingleReturnParser.allowed_ast_node:
                    raise SyntaxError(f"{node.__name__} is not allowed for single return")
        # case 2 for loop
        else:
            self.kernel_parser = ForLoopParser

        # else:
        #    raise SyntaxError("right now kernel func only support two patterns."
        #                      " 1st is return a single line. 2nd is a simple for loop")

    def check_return(self, node: ast.FunctionDef) -> Any:
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
        nd_ctx = NDArrayNode(
            self.return_var_name,
            self.kernel_p.return_types,
            self.shape_symbol_table,
            span)
        self.ndarray_context_table[self.return_var_name] = nd_ctx
        self.return_ctx = nd_ctx

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.visited_FunctionDef:
            raise SyntaxError("nested function def is not allowed.")
        self.visited_FunctionDef = True
        self.init_args(node)
        self.check_body(node)
        self.check_return(node)
