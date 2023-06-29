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
from dataclasses import dataclass
import matx.kernel.graphIR as _gir
from typing import Any, List, Union, TYPE_CHECKING, Dict
import matx.kernel.typing.utils as typing_utils
from .kernel_parser import KernelInspector

DTYPE_TO_MLIR = {
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "intc": "i32",
    "uint8": "i8",
    "uint16": "i16",
    "uint32": "i32",
    "uint64": "i64",
    "uintc": "i32",
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "longlong": "i64",
    "ulonglong": "i64"
}


@dataclass
class Ast2MLIR:
    arithmetic_binop: Dict[type, str]
    unaryop: Dict[type, str]
    boolop: Dict[type, str]


AST_TO_MLIR_PREFIX = Ast2MLIR(
    {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "",
        ast.Div: "",
        ast.FloorDiv: "",
        ast.Mod: "",
        ast.BitOr: "",
        ast.BitAnd: "",
        ast.BitXor: ""
    },
    {
        ast.USub: "",
        ast.Invert: "",
        ast.Not: ""
    },
    {
        ast.Gt: "",
        ast.GtE: "",
        ast.Lt: "",
        ast.LtE: "",
        ast.Eq: "",
        ast.NotEq: "",
        ast.Is: "",
        ast.IsNot: "",
        ast.And: "",
        ast.Or: ""}
)


class IrPrinter:
    def __init__(self):
        self.output = ''
        self._scope_indent = ""
        self._apply_indent = True

    def print(self, *args, sep=' ', end='\n'):
        text = sep.join(str(arg) for arg in args) + end
        if self._apply_indent:
            self.output += self._scope_indent
        self.output += text
        self._apply_indent = end == '\n'

    def new_scope(self):
        self._scope_indent += '\t'

    def pop_scope(self):
        if len(self._scope_indent) == 0:
            return
        self._scope_indent = self._scope_indent[:-1]

    def __str__(self):
        return self.output


def is_scalar_op(inputs: List[_gir.Tensor]) -> bool:
    return all((len(t.shape()) == 0 for t in inputs))


class GraphIRPrinter:

    def __init__(self, kernel_p: 'KernelInspector'):
        self.kernel_p = kernel_p

        self.graph_input: List[_gir.Node] = kernel_p.graph_input
        self.graph_output: List[_gir.Node] = kernel_p.graph_output
        self.graph_nodes: List[_gir.Node] = kernel_p.graph_nodes

        self.mlir_printer = IrPrinter()
        self._mlir_var_map = {}
        self._var_index_var = 0

    @property
    def var_index(self):
        i = self._var_index_var
        self._var_index_var += 1
        return f"%{i}"

    @staticmethod
    def _convert_type_to_mlir(node: _gir.Node):
        if isinstance(node, _gir.Scalar):
            return DTYPE_TO_MLIR[node.dtype()]
        if isinstance(node, _gir.Tensor):
            def dim_cvt(dim):
                if isinstance(dim, _gir.IntVar):
                    return "?x"
                if isinstance(dim, _gir.IntImm):
                    return f"{dim.value()}x"
                raise SyntaxError(f"not supported type {type(dim)} for {dim} as tensor dim")

            dims = (dim_cvt(d) for d in node.shape())
            return f"memref<{''.join(dims)}{DTYPE_TO_MLIR[node.dtype()]}>"
        if isinstance(node, (_gir.IntImm, _gir.IntVar)):
            return "i64"
        raise SyntaxError(f"Type {type} does not have a corresponding mlir type")

    def as_linalg_text(self):
        mlir_args = []
        mlir_arg_types = []
        for arg, node in self.kernel_p.arg_context_table.items():
            if not (isinstance(node, _gir.Tensor) or isinstance(node, _gir.Scalar)):
                raise NotImplementedError("func parameters can only be marked as ndarray or scalar")
            name = f"%{arg}"
            mlir_args.append(name)
            mlir_arg_types.append(self._convert_type_to_mlir(node))
            self._mlir_var_map[node] = name

        for dim, dim_var in self.kernel_p.shape_symbol_table.items():
            name = f"%{dim}"
            mlir_args.append(name)
            mlir_arg_types.append(self._convert_type_to_mlir(dim_var))
            self._mlir_var_map[dim_var] = name

        # todo assert one return
        if len(self.graph_output) != 1:
            raise SyntaxError(f"Expect the graph has exact 1 output "
                              f"but get {len(self.graph_output)}, "
                              f"which contains {self.graph_output}")

        self.mlir_printer.print(f"func.func @{self.kernel_p.func_name}", end='')
        mlir_name_and_type = (f"{name}: {t}" for name, t in zip(mlir_args, mlir_arg_types))
        self.mlir_printer.print(f"({', '.join(mlir_name_and_type)})", end='')
        if self.kernel_p.is_scalar_return():
            self.mlir_printer.print(
                f"->{self._convert_type_to_mlir(self.kernel_p.return_ctx)}", end='')
        self.mlir_printer.print("{")
        self.mlir_printer.new_scope()

        # convert graph
        self._visit(self.graph_output[0])
        if not self.kernel_p.is_scalar_return():
            self.mlir_printer.print(f"func.return")
        else:
            self.mlir_printer.print(
                f"func.return {self._mlir_var_map[self.graph_output[0]]}", end='')
            self.mlir_printer.print(f" : {self._convert_type_to_mlir(self.graph_output[0])}")
        self.mlir_printer.pop_scope()
        self.mlir_printer.print("}")
        return str(self.mlir_printer)

    def _generic_visit(self, node):
        """Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """
        raise NotImplementedError(f'This node is not supported now: {node}')

    def _visit(self, node: _gir.Node):
        method = "_visit_" + node.__class__.__name__
        visitor = getattr(self, method, self._generic_visit)
        visit_res = visitor(node)
        return visit_res

    def _visit_Scalar(self, node: _gir.Scalar):
        if node in self._mlir_var_map:
            return
        if node.is_a_const_num():
            mlir_type = DTYPE_TO_MLIR[node.dtype()]
            mlir_var = self.var_index
            stmt = f"{mlir_var} = arith.constant {node.value()} : {mlir_type}"
            self.mlir_printer.print(stmt)
            self._mlir_var_map[node] = mlir_var
        else:
            self._visit_Tensor(node)

    def _visit_Tensor(self, node: _gir.Tensor):
        if node in self._mlir_var_map:
            return
        src_ops = node.src_ops()
        src_op_list = list(src_ops)
        self._visit(src_op_list[0])
        self._mlir_var_map[node] = self._mlir_var_map[src_op_list[0]]

    def _gen_mlir_arith_statement(self, node, op, lhs, rhs, result_type):
        self._visit(lhs)
        self._visit(rhs)
        mlir_result_type = DTYPE_TO_MLIR[result_type]
        lhs_mlir_var_name = self._mlir_var_map[lhs]
        rhs_mlir_var_name = self._mlir_var_map[rhs]
        mlir_var = self.var_index
        mlir_op_name = AST_TO_MLIR_PREFIX.arithmetic_binop[op]
        stmt = f"{mlir_var} = arith.{mlir_op_name}{mlir_result_type[0]} " \
               f"{lhs_mlir_var_name}, {rhs_mlir_var_name} : " \
               f"{mlir_result_type}"
        self.mlir_printer.print(stmt)
        self._mlir_var_map[node] = mlir_var

    def _visit_BinaryElementWiseOperator(self, node: _gir.BinaryElementWiseOperator):
        inputs = node.get_inputs()
        lhs, rhs = inputs
        op = node.op_types[0]

        if is_scalar_op(inputs):
            self._gen_mlir_arith_statement(node, op, lhs, rhs, node.result_dtype)
        else:
            ...

    def _visit_CopyOperator(self, node: _gir.CopyOperator):
        copy_from = node.copy_from
        self._visit(copy_from)
        self._mlir_var_map[node] = self._mlir_var_map[copy_from]
