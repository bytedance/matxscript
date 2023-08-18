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
from functools import partial
from typing import List, Union, Dict, Callable, TYPE_CHECKING

import matx.kernel.graphIR as _gir
import matx.kernel.typing.utils as typing_utils
from .linalg_printer import LinalgGenericPrinter, LinalgReductionPrinter

if TYPE_CHECKING:
    from ..kernel_parser import FunctionParser


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
    "bool": MLIRType("i", 1),
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
        text = sep.join(str(arg) for arg in args)
        split_text = text.split("\n")
        for idx, s in enumerate(split_text):
            if self._apply_indent:
                self.output += self._scope_indent
            if idx != len(split_text) - 1:
                self.output += s + "\n"
            else:
                self.output += s + end
            self._apply_indent = True
        self._apply_indent = end == '\n'

    def new_scope(self):
        self._scope_indent += '\t'

    def pop_scope(self):
        if len(self._scope_indent) == 0:
            return
        self._scope_indent = self._scope_indent[:-1]

    def __str__(self):
        return self.output


def dim_cvt(dim):
    if isinstance(dim, int):
        return f"{dim}x"
    if isinstance(dim, _gir.IntVar):
        return "?x"
    if isinstance(dim, _gir.IntImm):
        return f"{dim.value()}x"
    if _gir.utils.is_graph_ir_scalar(dim):
        return "?x"
    raise SyntaxError(f"not supported type {type(dim)} for {dim} as tensor dim")


def is_scalar_op(inputs: List[_gir.Tensor]) -> bool:
    return all((len(t.shape()) == 0 for t in inputs))


def not_supported_op(*_, node=None):
    if node is None:
        raise NotImplementedError("calling op not supported")
    else:
        raise NotImplementedError(f"{node} of {type(node)} is not supported")


class GraphIRPrinter:

    def __init__(self, func_parser: 'FunctionParser'):
        self.func_parser = func_parser

        self.graph_input: List[_gir.Node] = func_parser.graph_input
        self.graph_output: List[_gir.Node] = func_parser.graph_output
        self.graph_nodes: List[_gir.Node] = func_parser.graph_nodes

        self.mlir_printer = IrPrinter()
        self.mlir_var_map = {}
        self.special_type_map = {}
        self._var_index_var = 0
        self.visited = set(self.graph_input)

        self.linalg_generic_map: Dict[_gir.ElementWiseOperator, LinalgGenericPrinter] = {}

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
            ast.USub: not_supported_op,
            ast.Invert: not_supported_op,
            ast.Not: self._gen_not_statement,
            ast.Gt: partial(
                self._gen_compare_statement,
                compare_type="gt"),
            ast.GtE: partial(
                self._gen_compare_statement,
                compare_type="ge"),
            ast.Lt: partial(
                self._gen_compare_statement,
                compare_type="lt"),
            ast.LtE: partial(
                self._gen_compare_statement,
                compare_type="le"),
            ast.Eq: partial(
                self._gen_compare_statement,
                compare_type="eq"),
            ast.NotEq: partial(
                self._gen_compare_statement,
                compare_type="ne"),
            ast.Is: not_supported_op,
            ast.IsNot: not_supported_op,
            ast.And: partial(self._gen_boolean_statement, bool_op_type="and"),
            ast.Or: partial(self._gen_boolean_statement, bool_op_type="or")}

    @property
    def new_var_name(self):
        i = self._var_index_var
        self._var_index_var += 1
        return f"%{i}"

    def get_strided_memref_type(self, node, shape=None, stride=None, offset=None) -> str:
        if node in self.special_type_map:
            return self.special_type_map[node]
        ndim = len(node.shape())
        if shape is None:
            mlir_shape = (dim_cvt(d) for d in node.shape())
        else:
            mlir_shape = (dim_cvt(d) for d in shape)
        return f"memref<{''.join(mlir_shape)}{DTYPE_TO_MLIR[node.dtype()]}, strided<[{', '.join(['?'] * ndim)}], offset: ?>>"

    def convert_type_to_mlir(self, node: _gir.Node):
        if node in self.special_type_map:
            return self.special_type_map[node]
        if isinstance(node, _gir.Scalar) or _gir.utils.is_graph_ir_scalar(node):
            return str(DTYPE_TO_MLIR[node.dtype()])
        if isinstance(node, _gir.Tensor):
            dims = (dim_cvt(d) for d in node.shape())
            return f"memref<{''.join(dims)}{DTYPE_TO_MLIR[node.dtype()]}>"
        if isinstance(node, (_gir.IntImm, _gir.IntVar)):
            return "i64"
        raise SyntaxError(f"Type {type(node)} does not have a corresponding mlir type")

    def as_linalg_text(self):
        mlir_args = []
        mlir_arg_types = []
        for arg, node in self.func_parser.arg_context_table.items():
            if not (isinstance(node, _gir.Tensor) or isinstance(node, _gir.Scalar)):
                raise NotImplementedError("func parameters can only be marked as ndarray or scalar")
            name = f"%{arg}"
            mlir_args.append(name)
            mlir_arg_types.append(self.convert_type_to_mlir(node))
            self.mlir_var_map[node] = name

        # todo assert one return
        if len(self.graph_output) != 1:
            raise SyntaxError(f"Expect the graph has exact 1 output "
                              f"but get {len(self.graph_output)}, "
                              f"which contains {self.graph_output}")

        self.mlir_printer.print(f"func.func @{self.func_parser.func_name}", end='')
        mlir_name_and_type = (f"{name}: {t}" for name, t in zip(mlir_args, mlir_arg_types))
        self.mlir_printer.print(f"({', '.join(mlir_name_and_type)})", end='')

        if self.func_parser.func_return_kind.is_scalar():
            rt_type = str(DTYPE_TO_MLIR[self.func_parser.return_dtype_str])
            self.mlir_printer.print(f"->{rt_type}", end='')
        elif self.func_parser.func_return_kind.is_dynamic_tensor():
            dims = (dim_cvt(d) for d in self.func_parser.return_shape)
            # todo use self.convert_type_to_mlir(self.graph_output[0])
            rt_type = f"memref<{''.join(dims)}{DTYPE_TO_MLIR[self.func_parser.return_dtype_str]}>"
            self.mlir_printer.print(f"->{rt_type}", end='')

        self.mlir_printer.print(" attributes {llvm.emit_c_interface} ", end='')
        self.mlir_printer.print("{")
        self.mlir_printer.new_scope()
        for dim, dim_var in self.func_parser.shape_symbol_table.items():
            self._get_symbol_value(dim, dim_var)

        # convert graph
        self.visit(self.graph_output[0], None)
        if self.func_parser.func_return_kind.is_scalar() or self.func_parser.func_return_kind.is_dynamic_tensor():
            self.mlir_printer.print(
                f"func.return {self.mlir_var_map[self.graph_output[0]]}", end='')
            self.mlir_printer.print(f" : {self.convert_type_to_mlir(self.graph_output[0])}")
        else:
            self.mlir_printer.print("func.return")
        self.mlir_printer.pop_scope()
        self.mlir_printer.print("}")
        return str(self.mlir_printer)


    def _get_symbol_value(self, dim, dim_var):
        corresponding_nd = None
        idx = -1
        for name, nd_node in self.func_parser.arg_context_table.items():
            if isinstance(nd_node, _gir.Tensor) and dim_var in nd_node.shape():
                corresponding_nd = nd_node
                idx = list(nd_node.shape()).index(dim_var)
        if corresponding_nd is None or idx == -1:
            raise SyntaxError("Symbol {} not found in arg_context_table".format(dim_var))

        name = f"%{dim}"
        nd_name = self.mlir_var_map[corresponding_nd]
        idx_name = self.new_var_name
        stmt0 = f"{idx_name} = arith.constant {idx} : index"
        stmt1 = f"{name}_idx = memref.dim {nd_name}, {idx_name} : {self.convert_type_to_mlir(corresponding_nd)}"
        stmt2 = f"{name} = index.castu {name}_idx : index to i64"
        self.mlir_printer.print(stmt0)
        self.mlir_printer.print(stmt1)
        self.mlir_printer.print(stmt2)
        self.mlir_var_map[dim_var] = name

    def _gen_arith_statement(
            self,
            node: _gir.BinaryElementWiseOperator,
            lhs: _gir.Scalar,
            rhs: _gir.Scalar,
            result_type: str,
            mlir_op_name: str,
            suffix_map=None):
        self.visit(lhs, node)
        self.visit(rhs, node)
        mlir_result_type = DTYPE_TO_MLIR[result_type]
        lhs_mlir_var_name = self.mlir_var_map[lhs]
        rhs_mlir_var_name = self.mlir_var_map[rhs]
        if lhs.dtype() != result_type:
            lhs_mlir_var_name = self._cast(lhs, result_type)
        if rhs.dtype() != result_type:
            rhs_mlir_var_name = self._cast(rhs, result_type)

        mlir_var = self.new_var_name
        suffix = mlir_result_type.code
        if suffix_map is not None and suffix in suffix_map:
            suffix = suffix_map[suffix]
        stmt = f"{mlir_var} = arith.{mlir_op_name}{suffix} " \
               f"{lhs_mlir_var_name}, {rhs_mlir_var_name} : " \
               f"{mlir_result_type}"
        self.mlir_printer.print(stmt)
        self.mlir_var_map[node] = mlir_var
        return mlir_var

    def _gen_floor_div_statement(self, node: _gir.BinaryElementWiseOperator,
                                 lhs: _gir.Scalar, rhs: _gir.Scalar, result_type: str):
        suffix_map = {"f": "divf",
                      "i": "floordivsi"}
        rt = self._gen_arith_statement(node, lhs, rhs, result_type, "", suffix_map)
        div_var = self.mlir_var_map[node]
        result_mlir_type = DTYPE_TO_MLIR[result_type]
        suffix_type = result_mlir_type.code
        if suffix_type != "f":
            return rt
        mlir_var = self.new_var_name
        stmt = f"{mlir_var} = math.floor {div_var} : {result_mlir_type}"
        self.mlir_printer.print(stmt)
        self.mlir_var_map[node] = mlir_var
        return mlir_var

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
            return self.mlir_var_map[operand]
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
        mlir_var = self.new_var_name
        stmt = f"{mlir_var} = {op} {self.mlir_var_map[operand]} : {origin_mlir_type} to {target_mlir_type}"
        self.mlir_printer.print(stmt)
        return mlir_var

    def _gen_mod_statement(
            self,
            node: _gir.BinaryElementWiseOperator,
            lhs: _gir.Scalar,
            rhs: _gir.Scalar,
            result_type: str):
        suffix_map = {"i": "si"}
        self.visit(lhs, node)
        self.visit(rhs, node)
        mlir_result_type = DTYPE_TO_MLIR[result_type]
        lhs_mlir_var_name = self.mlir_var_map[lhs]
        rhs_mlir_var_name = self.mlir_var_map[rhs]
        if lhs.dtype() != result_type:
            lhs_mlir_var_name = self._cast(lhs, result_type)
        if rhs.dtype() != result_type:
            rhs_mlir_var_name = self._cast(rhs, result_type)

        suffix = mlir_result_type.code
        mode_suffix = suffix
        if suffix_map is not None and mode_suffix in suffix_map:
            mode_suffix = suffix_map[mode_suffix]
        mod_op = f"arith.rem{mode_suffix}"
        mod1_var = self.new_var_name
        mod1 = f"{mod1_var} = {mod_op} {lhs_mlir_var_name}, {rhs_mlir_var_name} : {mlir_result_type}"
        add_var = self.new_var_name
        add = f"{add_var} = arith.add{suffix} {mod1_var}, {rhs_mlir_var_name} : {mlir_result_type}"
        mod2_var = self.new_var_name
        mod2 = f"{mod2_var} = {mod_op} {add_var}, {rhs_mlir_var_name} : {mlir_result_type}"
        self.mlir_printer.print(mod1)
        self.mlir_printer.print(add)
        self.mlir_printer.print(mod2)
        self.mlir_var_map[node] = mod2_var
        return mod2_var

    def _gen_compare_statement(self, node: _gir.BinaryElementWiseOperator, lhs: _gir.Scalar,
                               rhs: _gir.Scalar, result_type: str, compare_type: str):
        self.visit(lhs, node)
        self.visit(rhs, node)
        lhs_mlir_var_name = self.mlir_var_map[lhs]
        rhs_mlir_var_name = self.mlir_var_map[rhs]
        lhs_type = self.convert_type_to_mlir(lhs)
        suffix = lhs_type[0]
        predicate = compare_type
        if suffix == "f":
            predicate = "o" + predicate
        elif compare_type != "eq" and compare_type != "ne":
            if suffix == "ui":
                warnings.warn(f"Encountered a unsupported type: {suffix}."
                              f" Will try to treat it as unsigned int")
                predicate = "u" + predicate
            elif suffix == "i":
                predicate = "s" + predicate
            else:
                warnings.warn(f"Enconuntered a unsuppoerted type: {suffix}."
                              f" Will try to treat it as signed int")
                predicate = "u" + predicate

        mlir_var = self.new_var_name

        stmt = f"{mlir_var} = arith.cmp{suffix} {predicate}, " \
               f"{lhs_mlir_var_name}, {rhs_mlir_var_name} : {lhs_type}"
        self.mlir_printer.print(stmt)
        self.mlir_var_map[node] = mlir_var
        return mlir_var

    def _gen_boolean_statement(self, node: _gir.BinaryElementWiseOperator, lhs: _gir.Scalar,
                               rhs: _gir.Scalar, result_type: str, bool_op_type: str):
        self.visit(lhs, node)
        self.visit(rhs, node)
        lhs_mlir_var_name = self.mlir_var_map[lhs]
        rhs_mlir_var_name = self.mlir_var_map[rhs]
        mlir_result_type = DTYPE_TO_MLIR[result_type]
        suffix = mlir_result_type.code

        mlir_var = self.new_var_name

        stmt = f"{mlir_var} = arith.{bool_op_type}{suffix}" \
               f" {lhs_mlir_var_name}, {rhs_mlir_var_name} : {mlir_result_type}"
        self.mlir_printer.print(stmt)
        self.mlir_var_map[node] = mlir_var
        return mlir_var

    def _gen_not_statement(self, node: _gir.UnaryElementWiseOperator, operand: _gir.Scalar,
                           result_type: str):
        self.visit(operand, node)
        operand_mlir_var_name = self.mlir_var_map[operand]
        mlir_result_type = DTYPE_TO_MLIR[result_type]
        operand_type = DTYPE_TO_MLIR[operand.dtype()]

        const_1 = self.new_var_name
        xor = self.new_var_name

        const_stmt = f"{const_1} = arith.constant 1 : {operand_type}"
        self.mlir_printer.print(const_stmt)
        xor_stmt = f"{xor} = arith.xori {const_1}, {operand_mlir_var_name} : {mlir_result_type}"
        self.mlir_printer.print(xor_stmt)
        self.mlir_var_map[node] = xor
        return xor

    def _gen_lingalg_generic(self, node: _gir.ElementWiseOperator, _from: _gir.Tensor):
        if node not in self.linalg_generic_map:
            self.linalg_generic_map[node] = LinalgGenericPrinter(node, self)

        lg_printer = self.linalg_generic_map[node]
        # allocate _from
        if _from not in self.mlir_var_map:
            mlir_var = self.new_var_name
            shape_var = (self.mlir_var_map[d] for d in _from.shape() if isinstance(d, _gir.IntVar))
            casted_shape_var = []
            for sv in shape_var:
                sv_name = self.new_var_name
                cast_stmt = f"{sv_name} = builtin.unrealized_conversion_cast {sv} : " \
                            f"i64 to index"
                self.mlir_printer.print(cast_stmt)
                casted_shape_var.append(sv_name)
            if _from in self.graph_output:
                alloc_op = "memref.alloc"
            else:
                alloc_op = "memref.alloca"
            stmt = f"{mlir_var} = {alloc_op}({', '.join(casted_shape_var)}) : {self.convert_type_to_mlir(_from)}"
            lg_printer.add_allocate_stmt((mlir_var, _from, stmt))
            self.mlir_var_map[_from] = mlir_var
        else:
            mlir_var = self.mlir_var_map[_from]
        lg_printer.add_output(_from, mlir_var)
        if lg_printer.ok_to_generic():
            for i in node.get_inputs():
                self.visit(i, node)
            lg_printer.gen_code()
        return mlir_var

    def _generic_visit(self, node, _from):
        raise NotImplementedError(f'This node is not supported now: {node.__class__.__name__}')

    def _recursive_visit(self, _class):
        method = "_visit_" + _class.__name__
        visitor: Callable = getattr(self, method, None)
        if visitor is not None:
            return visitor
        for base in _class.__bases__:
            rt_method = self._recursive_visit(base)
            if rt_method is not None:
                return rt_method
        return None

    def visit(self, node: _gir.Node, _from: Union[_gir.Node, None]):
        visitor: Callable = self._recursive_visit(node.__class__)
        if visitor is None:
            visitor = self._generic_visit
        visit_res = visitor(node, _from)
        return visit_res

    def _visit_IntImm(self, node: _gir.IntImm, _from: _gir.Node) -> str:
        if node in self.visited:
            return self.mlir_var_map[node]
        mlir_var = self.new_var_name
        stmt = f"{mlir_var} = index.constant {node.value()}"
        self.mlir_printer.print(stmt)
        self.mlir_var_map[node] = mlir_var
        self.visited.add(node)
        return mlir_var

    def _convert_index(self, index, node):
        result = []
        for e in index:
            if _gir.utils.is_graph_ir_scalar(e):
                self.visit(e, node)
                result.append(self._cast_to_idx(e))
            elif isinstance(e, _gir.IntImm):
                result.append(self.visit(e, node))
            else:
                raise SyntaxError("not support")
        return result

    def _visit_TensorGetItemOperator(
            self,
            node: _gir.TensorGetItemOperator,
            _from: _gir.Node) -> str:
        result_name = self.new_var_name
        tensor_name = self._visit_Tensor(node.tensor, node)
        tensor_type = self.convert_type_to_mlir(node.tensor)
        index = self._convert_index(node.index, node)
        stmt = f"{result_name} = memref.load {tensor_name}[{', '.join(index)}] : {tensor_type}"
        self.mlir_printer.print(stmt)
        return result_name

    def _visit_TensorSliceOperator(self, node: _gir.TensorSliceOperator, _from: _gir.Node) -> str:
        def shape_helper(l):
            result = []
            for e in l:
                if _gir.utils.is_graph_ir_scalar(e):
                    self.visit(e, node)
                    result.append(self._cast_to_idx(e))
                elif isinstance(e, _gir.IntImm):
                    rc = self.visit_int_var(e)
                    result.append(rc)
                elif isinstance(e, _gir.IntVar):
                    self.visit_int_var(e)
                    result.append(self._cast_to_idx(e))
                elif isinstance(e, int):
                    result.append(e)
                else:
                    raise SyntaxError(f"not supported {type(e)}")
            return result

        result_name = self.new_var_name
        tensor = self.visit(node.tensor, node)
        offset = ', '.join(map(str, shape_helper(node.offset)))
        size = ', '.join(map(str, shape_helper(node.size)))
        stride = ', '.join(map(str, shape_helper(node.stride)))
        o_type = self.convert_type_to_mlir(node.tensor)
        v_type = self.get_strided_memref_type(_from, node.shape)
        self.special_type_map[_from] = v_type
        stmt = f"{result_name} = memref.subview {tensor} [{offset}][{size}][{stride}] : {o_type} to {v_type}"
        self.mlir_printer.print(stmt)
        return result_name

    def _visit_Scalar(self, node: _gir.Scalar, _from: _gir.Node) -> str:
        if node in self.visited:
            return self.mlir_var_map[node]
        if node.is_a_const_num():
            mlir_type = DTYPE_TO_MLIR[node.dtype()]
            mlir_var = self.new_var_name
            stmt = f"{mlir_var} = arith.constant {node.value()} : {mlir_type}"
            self.mlir_printer.print(stmt)
            self.mlir_var_map[node] = mlir_var
            self.visited.add(node)
            return mlir_var
        else:
            return self._visit_Tensor(node, _from)

    def _visit_Tensor(self, node: _gir.Tensor, _from: _gir.Node) -> str:
        if node in self.visited:
            return self.mlir_var_map[node]
        self.visited.add(node)
        src_ops = node.src_ops()
        src_op_list = list(src_ops)
        mlir_name = self.visit(src_op_list[0], node)
        self.mlir_var_map[node] = mlir_name
        return mlir_name

    def _visit_FusedElementWiseOperator(
            self,
            node: _gir.BinaryElementWiseOperator,
            _from: _gir.Node) -> str:
        inputs = node.get_inputs()
        if is_scalar_op(inputs):
            raise SyntaxError("fused op should not be a scalar op")
        return self._gen_lingalg_generic(node, _from)

    def _visit_BinaryElementWiseOperator(
            self,
            node: _gir.BinaryElementWiseOperator,
            _from: _gir.Node) -> str:
        inputs = node.get_inputs()
        lhs, rhs = inputs
        op = node.op_types[0]

        if is_scalar_op(inputs):
            visitor: Callable = self.op[op]
            return visitor(node, lhs, rhs, node.result_dtype)
        else:
            return self._gen_lingalg_generic(node, _from)

    def _visit_UnaryElementWiseOperator(
            self,
            node: _gir.UnaryElementWiseOperator,
            _from: _gir.Node) -> str:
        inputs = node.get_inputs()
        operand = inputs[0]
        op = node.op_types[0]

        if is_scalar_op(inputs):
            visitor: Callable = self.op[op]
            return visitor(node, operand, node.result_dtype)
        else:
            return self._gen_lingalg_generic(node, _from)

    def _visit_CopyOperator(self, node: _gir.CopyOperator, _from: _gir.Node) -> str:
        copy_from = node.copy_from
        copy_to = node.copy_to
        self.visit(copy_from, _from)
        if copy_to.dtype() != copy_from.dtype():
            casted_copy_from_mlir_var_name = self._cast(copy_from, copy_to.dtype())
            self.mlir_var_map[node] = casted_copy_from_mlir_var_name
            return casted_copy_from_mlir_var_name
        else:
            self.mlir_var_map[node] = self.mlir_var_map[copy_from]
            return self.mlir_var_map[copy_from]

    def _visit_DeepCopyOperator(self, node: _gir.DeepCopyOperator, _from: _gir.Node) -> str:
        copy_from = node.copy_from
        copy_to = node.copy_to
        self.visit(copy_from, node)
        stmt = f"linalg.copy \n\tins({self.mlir_var_map[copy_from]}: {self.convert_type_to_mlir(copy_from)})" \
               f"\n\touts({self.mlir_var_map[copy_to]}: {self.convert_type_to_mlir(copy_to)})"
        self.mlir_printer.print(stmt)
        return self.mlir_var_map[copy_to]

    def _visit_ReductionOperator(self, node: _gir.ReductionOperator, _from: _gir.Node) -> str:
        for init_value in node.sub_graph_init_values:
            self.visit(init_value, node)
        reduction_printer = LinalgReductionPrinter(node, self)
        reduction_printer.gen_code()
        return reduction_printer.results[0]

    def _visit_TensorSetItemOperator(
            self,
            node: _gir.TensorSetItemOperator,
            _from: _gir.Node) -> str:
        tensor_mlir_name = self.visit(node.tensor, node)
        _ = self.visit(node.value, node)
        tensor_type = self.convert_type_to_mlir(node.tensor)
        index = self._convert_index(node.index, node)
        value_mlie_name = self._cast(node.value, node.tensor.dtype())
        stmt = f"memref.store {value_mlie_name}, {tensor_mlir_name}[{', '.join(index)}] : {tensor_type}"
        self.mlir_printer.print(stmt)
        return tensor_mlir_name

    def _cast_to_idx(self, node):
        node_var = self.mlir_var_map[node]
        index_var = self.new_var_name
        in_type = self.convert_type_to_mlir(node)
        if in_type != "i64":
            node_var = self._cast(node, "int64")
        stmt = f"{index_var} = arith.index_cast {node_var} : i64 to index"
        self.mlir_printer.print(stmt)
        return index_var

    def visit_int_var(self, node):
        if isinstance(node, _gir.IntImm):
            mlir_name = self.new_var_name
            stmt = f"{mlir_name} = index.constant {node.value()}"
            self.mlir_printer.print(stmt)
            return mlir_name
        if node in self.mlir_var_map:
            return self.mlir_var_map[node]
        raise NotImplementedError("not supported now")
