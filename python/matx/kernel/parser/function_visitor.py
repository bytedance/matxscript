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
from typing import Any, Dict, List, Union, TYPE_CHECKING

import numpy as np

import matx.kernel.graphIR as _gir
import matx.kernel.symbol.utils as symbol_utils
import matx.kernel.typing.utils as typing_utils
from matx.kernel.func_registery import FUNC_REGISTRY
from matx.kernel.parser.general_parser import GeneralParser
from matx.kernel.parser.loop_parser import LoopParser
from matx.kernel.parser.tensor_op_parser import TensorOpParser
from matx.kernel.typing import NDArrayType as kernelNDArrayT
from matx.script import context as script_context

from .utils import BodyIterator, FuncReturnKind

if TYPE_CHECKING:
    from matx.kernel.kernel_parser import KernelParser


class FunctionVisitor(ast.NodeVisitor):
    return_var_id = 93502842947314

    def __init__(
            self,
            kernel_p: 'KernelParser',
            node: script_context.ASTNode,
            inline=True):

        self.return_var_name = f'__return_{FunctionVisitor.return_var_id}__'
        FunctionVisitor.return_var_id += 1
        self.kernel_p = kernel_p

        # necessary for reuse script functionality
        self.root_node = node
        self.context = None

        # for kernel use
        self.arg_context_table: Dict[str, _gir.Tensor] = {}
        self.shape_symbol_table: Dict[str, _gir.IntVar] = {}
        self.tmp_scalar_table: Dict[str, _gir.Scalar] = {}
        self.tmp_ndarray_table: Dict[str, _gir.Tensor] = {}
        self.return_ctx: Union[None, _gir.Tensor] = None
        self.return_types = kernel_p.return_types
        self.return_dtype_str: str = ""
        self.return_shape = []

        if self.kernel_p.empty_return_signature:
            self.func_return_kind: FuncReturnKind = FuncReturnKind.VOID
        elif typing_utils.is_scalar(self.return_types):
            self.func_return_kind: FuncReturnKind = FuncReturnKind.SCALAR
        elif typing_utils.is_dynamic_ndarray(self.return_types):
            self.func_return_kind: FuncReturnKind = FuncReturnKind.DYNAMIC_TENSOR
        else:
            self.func_return_kind: FuncReturnKind = FuncReturnKind.STATIC_TENSOR

        self.func_name: str = kernel_p.func_name

        # for checking
        self.visited_FunctionDef = False  # nested function definition is not allowed
        self.ast_nodes = []

        # for graph IR
        self.graph_input: List[_gir.Node] = []
        self.graph_output: List[_gir.Node] = []
        self.graph_nodes: List[_gir.Node] = []

        self.body_visitor = None
        self.can_inline = inline

        self.tmp_var_id = 891371

    def get_new_tmp_var_id(self):
        self.tmp_var_id += 1
        return f"__{self.tmp_var_id}_"

    def check_and_dispatch(self, node: ast.AST) -> Any:
        if isinstance(node, ast.For):
            p = LoopParser(self)
        # elif TensorOpParser.can_parse(self, node):
        #     p = TensorOpParser(self)
        else:
            p = GeneralParser(self)
        t = p.visit(node)
        self.can_inline = self.can_inline and p.can_inline
        return t

    def declare_shape_var(self, type_annotation):
        if not isinstance(type_annotation, kernelNDArrayT):
            return
        shape_symbols = []
        for dim in type_annotation.shape:
            if not typing_utils.is_symbol(dim):
                continue
            if str(dim) in self.shape_symbol_table:
                continue
            sym_ctx = _gir.graph.IntVar([0, np.iinfo(np.int64).max], symbolic_value=dim)
            self.shape_symbol_table[str(dim)] = sym_ctx
            self.graph_input.append(sym_ctx)
            self.graph_nodes.append(sym_ctx)
            shape_symbols.append(sym_ctx)
        return shape_symbols

    def init_args(self) -> None:
        for arg, type_annotation in self.kernel_p.args.items():
            if typing_utils.is_scalar_type(type_annotation):
                dtype = typing_utils.convert_to_string_dtype(type_annotation.dtype)
                scalar_ctx = _gir.Scalar(name=arg, dtype=dtype, is_input=True)
                self.arg_context_table[arg] = scalar_ctx
                self.graph_input.append(scalar_ctx)
                self.graph_nodes.append(scalar_ctx)
            elif typing_utils.is_ndarray_type(type_annotation):
                self.declare_shape_var(type_annotation)
                dtype = typing_utils.convert_to_string_dtype(type_annotation.dtype)
                shape = self.convert_to_gir_shape(type_annotation.shape)
                nd_ctx = _gir.Tensor(shape, name=arg, dtype=dtype, is_input=True)
                self.arg_context_table[arg] = nd_ctx
                self.graph_input.append(nd_ctx)
                self.graph_nodes.append(nd_ctx)
            else:
                raise SyntaxError(f"right now only kernel ndarray are supported, "
                                  f"but get {type_annotation} for {arg}")

    def make_return(self, shape, dtype: str):
        if typing_utils.is_ndarray_type(self.kernel_p.return_types):
            nd_ctx = _gir.Tensor(
                shape=shape,
                name=self.return_var_name,
                dtype=dtype,
                is_input=True,
                is_output=True)

            new_arg_context_table = {self.return_var_name: nd_ctx, **self.arg_context_table}

            self.arg_context_table = new_arg_context_table
            self.return_ctx = nd_ctx
        else:
            raise SyntaxError("kernel function is supposed to return a kernel ndarray"
                              " or scalar(a.k.a kernel ndarray with shape 1)"
                              f"but get {self.kernel_p.return_types}")
        self.graph_nodes.append(self.return_ctx)
        self.graph_output.append(self.return_ctx)

    def check_return(self) -> Any:
        # todo fix return input ndarray
        if self.func_return_kind.is_void():
            return
        if self.func_return_kind.is_scalar():
            dtype = typing_utils.convert_to_string_dtype(self.kernel_p.return_types.dtype)
            self.return_dtype_str = dtype
            self.return_shape = []
            return
        if not typing_utils.is_ndarray_type(self.kernel_p.return_types):
            raise SyntaxError("kernel function is supposed to return a kernel ndarray"
                              " or scalar(a.k.a kernel ndarray with shape 1)"
                              f"but get {self.kernel_p.return_types}")
        for dim in self.kernel_p.return_types.shape:
            if typing_utils.is_symbol(dim) and (not self.all_in_dict(dim)):
                raise SyntaxError(
                    f"{dim} in the return type is not defined in any of the shape in input args")
        if self.return_var_name in self.arg_context_table:
            raise SyntaxError("return var name is being used")

        dtype = typing_utils.convert_to_string_dtype(self.kernel_p.return_types.dtype)
        self.return_dtype_str = dtype
        self.return_shape = self.convert_to_gir_shape(self.kernel_p.return_types.shape)
        if self.func_return_kind.is_static_tensor():
            self.make_return(self.return_shape, self.return_dtype_str)

    def parse_body(self, auto_add_return=False):
        self.body_visitor = BodyIterator(self.context.node_stack, auto_add_return)
        while self.body_visitor.has_next():
            res = self.check_and_dispatch(self.body_visitor.next())
            self.body_visitor.push_ir(res)
        return self.body_visitor.body

    def continue_parse_as_scoped(self):
        ir_sofar = self.body_visitor.body
        self.body_visitor.body = []
        while self.body_visitor.has_next():
            res = self.check_and_dispatch(self.body_visitor.next())
            self.body_visitor.push_ir(res)
        scoped_ir = self.body_visitor.body
        self.body_visitor.body = ir_sofar
        return scoped_ir

    def visit_body(self, node: ast.FunctionDef):
        self.context = script_context.ScopeContext()
        self.context.new_scope(nodes=node.body)
        self.parse_body(True)
        self.context.pop_scope()
        # _gir.utils.draw_graph(self.graph_nodes)
        cfuser = _gir.graph_pass.TmpVarEliminator()
        cfuser.apply(self.graph_input, self.graph_output, self.graph_nodes)
        # _gir.utils.draw_graph(self.graph_nodes)
        efuser = _gir.graph_pass.ElementWiseOpFuser()
        efuser.apply(self.graph_input, self.graph_output, self.graph_nodes)
        # _gir.utils.draw_graph(self.graph_nodes)
        udeleter = _gir.graph_pass.UnreachableNodeEliminator()
        udeleter.apply(self.graph_input, self.graph_output, self.graph_nodes)
        # _gir.utils.draw_graph(self.graph_nodes)
        return self

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.visited_FunctionDef:
            raise SyntaxError("nested function def is not allowed.")
        FUNC_REGISTRY[id(self.kernel_p.func)] = self
        self.visited_FunctionDef = True
        self.init_args()
        self.check_return()
        return self.visit_body(node)

    def substitute_symbol_expr(self, symbol_expr):
        free_symbols = symbol_expr.free_symbols
        sub_dict = {}
        for s in free_symbols:
            if str(s) in self.shape_symbol_table:
                sub_dict[str(s)] = self.shape_symbol_table[str(s)]
            else:
                raise SyntaxError(f"the symbol {s} in the expr ({symbol_expr}) is not defined.")
        return symbol_expr.subs(sub_dict).evalf()

    def convert_to_gir_shape(self, shape):

        gir_shape = []
        for d in shape:
            if symbol_utils.is_symbol_expression(d):
                _ = self.substitute_symbol_expr(d)
            elif symbol_utils.is_symbol(d):
                if str(d) in self.shape_symbol_table:
                    gir_shape.append(self.shape_symbol_table[str(d)])
                else:
                    raise SyntaxError(f"the symbol in the shape ({shape}) is not defined yet.")
            elif isinstance(d, int):
                node = _gir.IntImm(d)
                gir_shape.append(node)
                self.graph_nodes.append(node)
            else:
                raise SyntaxError(
                    f"the shape ({shape}) of the ndarray is expected to be an symbol or int,"
                    f" but get {d} ({type(d)})")
        return gir_shape

    def all_in_dict(self, expr):
        return all(str(s) in self.shape_symbol_table for s in expr.free_symbols)
