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
import warnings
from dataclasses import dataclass
import matx.kernel.graphIR as _gir
from typing import Any, List, Union, TYPE_CHECKING, Dict, Callable
import matx.kernel.typing.utils as typing_utils
from functools import partial
from .kernel_parser import KernelInspector


@dataclass
class MLIRType:
    true_code: str
    bits: int

    @property
    def code(self):
        return self.true_code if self.true_code != "ui" else "i"

    def __str__(self):
        return f"{self.code}{self.bits}"


DTYPE_TO_MLIR = {
    "int8": MLIRType("i", 8),
    "int16": MLIRType("i", 16),
    "int32": MLIRType("i", 32),
    "int64": MLIRType("i", 64),
    "intc": MLIRType("i", 32),
    "uint8": MLIRType("ui", 8),
    "uint16": MLIRType("ui", 16),
    "uint32": MLIRType("ui", 32),
    "uint64": MLIRType("ui", 64),
    "uintc": MLIRType("ui", 32),
    "float16": MLIRType("f", 16),
    "float32": MLIRType("f", 32),
    "float64": MLIRType("f", 64),
    "longlong": MLIRType("i", 64),
    "ulonglong": MLIRType("ui", 64)
}


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


def not_supported_op(*args, node=None):
    if node is None:
        raise NotImplementedError("calling op not supported")
    else:
        raise NotImplementedError(f"{node} of {type(node)} is not supported")


class GraphIRPrinter:

    def __init__(self, kernel_p: 'KernelInspector'):
        self.kernel_p = kernel_p

        self.graph_input: List[_gir.Node] = kernel_p.graph_input
        self.graph_output: List[_gir.Node] = kernel_p.graph_output
        self.graph_nodes: List[_gir.Node] = kernel_p.graph_nodes

        self.mlir_printer = IrPrinter()
        self._mlir_var_map = {}
        self._var_index_var = 0

        self.op = {
            ast.Add: partial(
                self._gen_arith_statement,
                mlir_op_name="add"),
            ast.Sub: partial(
                self._gen_arith_statement,
                mlir_op_name="sub"),
            ast.Mult: partial(
                self._gen_arith_statement,
                mlir_op_name="mul"),
            ast.Div: partial(
                self._gen_arith_statement,
                mlir_op_name="",
                suffix_map={
                    "f": "divf",
                    "i": "floordivsi"}),
            ast.FloorDiv: self._gen_floor_div_statement,
            ast.Mod: self._gen_mod_statement,
            ast.BitOr: not_supported_op,
            ast.BitAnd: not_supported_op,
            ast.BitXor: not_supported_op,
            ast.USub: None,
            ast.Invert: None,
            ast.Not: None,
            ast.Gt: partial(
                self._gen_comapre_statement,
                compare_type="gt"),
            ast.GtE: partial(
                self._gen_comapre_statement,
                compare_type="ge"),
            ast.Lt: partial(
                self._gen_comapre_statement,
                compare_type="lt"),
            ast.LtE: partial(
                self._gen_comapre_statement,
                compare_type="le"),
            ast.Eq: partial(
                self._gen_comapre_statement,
                compare_type="eq"),
            ast.NotEq: partial(
                self._gen_comapre_statement,
                compare_type="ne"),
            ast.Is: not_supported_op,
            ast.IsNot: not_supported_op,
            ast.And: None,
            ast.Or: None}

    @property
    def var_index(self):
        i = self._var_index_var
        self._var_index_var += 1
        return f"%{i}"

    @staticmethod
    def _convert_type_to_mlir(node: _gir.Node):
        if isinstance(node, _gir.Scalar):
            return str(DTYPE_TO_MLIR[node.dtype()])
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

    def _gen_arith_statement(self, node: _gir.BinaryElementWiseOperator, lhs: _gir.Scalar, rhs: _gir.Scalar,
                             result_type: str, mlir_op_name: str, suffix_map=None):
        self._visit(lhs)
        self._visit(rhs)
        mlir_result_type = DTYPE_TO_MLIR[result_type]
        lhs_mlir_var_name = self._mlir_var_map[lhs]
        rhs_mlir_var_name = self._mlir_var_map[rhs]
        if lhs.dtype() != result_type:
            lhs_mlir_var_name = self._cast(lhs, result_type)
        if rhs.dtype() != result_type:
            rhs_mlir_var_name = self._cast(rhs, result_type)

        mlir_var = self.var_index
        suffix = mlir_result_type.code
        if suffix_map is not None and suffix in suffix_map:
            suffix = suffix_map[suffix]
        stmt = f"{mlir_var} = arith.{mlir_op_name}{suffix} " \
               f"{lhs_mlir_var_name}, {rhs_mlir_var_name} : " \
               f"{mlir_result_type}"
        self.mlir_printer.print(stmt)
        self._mlir_var_map[node] = mlir_var

    def _gen_floor_div_statement(self, node: _gir.BinaryElementWiseOperator,
                                 lhs: _gir.Scalar, rhs: _gir.Scalar, result_type: str):
        suffix_map = {"f": "divf",
                      "i": "floordivsi"}
        self._gen_arith_statement(node, lhs, rhs, result_type, "", suffix_map)
        div_var = self._mlir_var_map[node]
        result_mlir_type = DTYPE_TO_MLIR[result_type]
        suffix_type = result_mlir_type.code
        if suffix_type != "f":
            return
        mlir_var = self.var_index
        stmt = f"{mlir_var} = math.floor {div_var} : {result_mlir_type}"
        self.mlir_printer.print(stmt)
        self._mlir_var_map[node] = mlir_var

    def _cast(self, operand: _gir.Scalar, target_type: str):
        """
         arith.extf - cast from floating-point to wider floating-point
         arith.extsi - integer sign extension operation
         arith.extui - integer zero extension operation
         arith.fptosi - cast from floating-point type to signed integer type
         arith.fptoui - cast from floating-point type to unsigned integer type
         arith.sitofp - cast from signed integer type to floating-point
         arith.truncf - cast from floating-point to narrower floating-point
         arith.trunci - integer truncation operation
         arith.uitofp - cast from unsigned integer type to floating-point
        """

        def compare_f(target, origin):
            return {(target < origin): 0, (target == origin): 1, (target > origin): 2}[True]

        origin_type = operand.dtype()
        if origin_type == target_type:
            return self._mlir_var_map[operand]
        origin_mlir_type = DTYPE_TO_MLIR[origin_type]
        target_mlir_type = DTYPE_TO_MLIR[target_type]
        origin_type_bits = origin_mlir_type.bits
        origin_type_code = origin_mlir_type.true_code
        target_type_bits = target_mlir_type.bits
        target_type_code = target_mlir_type.true_code
        compare = compare_f(target_type_bits, origin_type_bits)

        op_map = {
            ("i", "i"): ["arith.trunci", "arith.bitcast", "arith.extsi"],
            ("i", "ui"): ["arith.trunci", "arith.bitcast", "arith.extui"],
            ("i", "f"): ["arith.sitofp", "arith.sitofp", "arith.sitofp"],
            ("ui", "i"): ["arith.trunci", "arith.bitcast", "arith.extui"],
            ("ui", "ui"): ["arith.trunci", "arith.bitcast", "arith.extui"],
            ("ui", "f"): ["arith.uitofp", "arith.uitofp", "arith.uitofp"],
            ("f", "i"): ["arith.fptosi", "arith.fptosi", "arith.fptosi"],
            ("f", "ui"): ["arith.fptoui", "arith.fptoui", "arith.fptoui"],
            ("f", "f"): ["arith.truncf", "arith.bitcast", "arith.extf"]
        }
        op = op_map[(origin_type_code, target_type_code)][compare]
        mlir_var = self.var_index
        stmt = f"{mlir_var} = {op} {self._mlir_var_map[operand]} : {origin_mlir_type} to {target_mlir_type}"
        self.mlir_printer.print(stmt)
        return mlir_var

    def _gen_mod_statement(self, node: _gir.BinaryElementWiseOperator, lhs: _gir.Scalar, rhs: _gir.Scalar,
                           result_type: str):
        suffix_map = {"i": "si"}
        self._gen_arith_statement(node, lhs, rhs, result_type, "rem", suffix_map)
        

    def _gen_comapre_statement(self, node: _gir.BinaryElementWiseOperator, compare_type: str, lhs: _gir.Scalar,
                               rhs: _gir.Scalar):
        self._visit(lhs)
        self._visit(rhs)
        lhs_mlir_var_name = self._mlir_var_map[lhs]
        rhs_mlir_var_name = self._mlir_var_map[rhs]
        lhs_type = self._convert_type_to_mlir(lhs)
        suffix = lhs_type[0]
        predicate = compare_type
        if suffix == "f":
            predicate = "o" + predicate
        elif compare_type != "eq" and compare_type != "ne":
            if suffix == "ui":
                warnings.warn(f"Enconuntered a unsuppoerted type: {suffix}."
                              f" Will try to treat it as unsigned int")
                predicate = "u" + predicate
            elif suffix == "i":
                predicate = "s" + predicate
            else:
                warnings.warn(f"Enconuntered a unsuppoerted type: {suffix}."
                              f" Will try to treat it as signed int")
                predicate = "u" + predicate

        mlir_var = self.var_index

        stmt = f"{mlir_var} = arith.cmp{suffix} {predicate}, " \
               f"{lhs_mlir_var_name}, {rhs_mlir_var_name} : {lhs_type}"
        self.mlir_printer.print(stmt)
        self._mlir_var_map[node] = mlir_var

    def _generic_visit(self, node):
        raise NotImplementedError(f'This node is not supported now: {node}')

    def _visit(self, node: _gir.Node):
        method = "_visit_" + node.__class__.__name__
        visitor: Callable = getattr(self, method, self._generic_visit)
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

    def _visit_BinaryElementWiseOperator(self, node: _gir.BinaryElementWiseOperator):
        inputs = node.get_inputs()
        lhs, rhs = inputs
        op = node.op_types[0]

        if is_scalar_op(inputs):
            visitor: Callable = self.op[op]
            visitor(node, lhs, rhs, node.result_dtype)
        else:
            ...

    def _visit_CopyOperator(self, node: _gir.CopyOperator):
        copy_from = node.copy_from
        self._visit(copy_from)
        self._mlir_var_map[node] = self._mlir_var_map[copy_from]

    def _visit_DeepCopyOperator(self, node: _gir.DeepCopyOperator):
        return self._generic_visit(node)
