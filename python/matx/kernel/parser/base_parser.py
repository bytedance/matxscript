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
import numbers
from typing import Any, List, Union, TYPE_CHECKING

import matx.kernel.graphIR as _gir
import matx.kernel.typing.utils as typing_utils
from .utils import scalar_or_int_var
import matx.kernel.kernel_parser
from matx.kernel.func_registery import FUNC_REGISTRY

if TYPE_CHECKING:
    from ..kernel_parser import KernelInspector


# todo update return
class BaseParser(ast.NodeVisitor):

    def __init__(
            self,
            kernel_p: 'KernelInspector'):
        self.kernel_p = kernel_p

        # necessary for reuse script functionality
        self.root_node = self.kernel_p.root_node

        # for kernel use
        self.arg_context_table = self.kernel_p.arg_context_table
        self.shape_symbol_table = self.kernel_p.shape_symbol_table
        self.tmp_scalar_table = self.kernel_p.tmp_scalar_table
        self.tmp_ndarray_table = self.kernel_p.tmp_ndarray_table
        self.return_ctx = self.kernel_p.return_ctx

        self.reads = []

        self.can_inline = True

    def generic_visit(self, node):
        """Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """
        raise NotImplementedError(f'This node is not supported now: {node}')

    def visit(self, node: Any) -> Any:
        """Override method in ast.NodeVisitor"""
        method = "visit_" + node.__class__.__name__
        print(method)
        visitor = getattr(self, method, self.generic_visit)
        visit_res = visitor(node)
        return visit_res

    def visit_Constant(self, node: ast.Constant) -> _gir.Tensor:
        if node.value is None:
            raise SyntaxError("None is not allowed")
        elif isinstance(node.value, numbers.Number):
            dtype = typing_utils.get_dtype_str(node.value)
            const_scalar_ctx = _gir.Scalar(value=node.value, dtype=dtype, is_internal_constant=True)
            self.kernel_p.graph_nodes.append(const_scalar_ctx)
            return const_scalar_ctx
        else:
            raise NotImplementedError(f'Unsupported value {node.value}')

    # variables
    def visit_Name(self, node: ast.Name) -> _gir.Node:
        if isinstance(node.ctx, ast.Del):
            raise SyntaxError(f"del {node.id} is not allowed")
        name = node.id
        if name in self.shape_symbol_table:
            ctx = self.shape_symbol_table[name]
            return ctx
        if name in self.arg_context_table:
            ctx = self.arg_context_table[name]
            return ctx
        if name in self.tmp_scalar_table:
            ctx = self.tmp_scalar_table[name]
            return ctx
        if name in self.tmp_ndarray_table:
            ctx = self.tmp_ndarray_table[name]
            return ctx
        raise SyntaxError(f'the type of {name} is not support')
        # return node.id

    # Expressions
    def visit_UnaryOp(self, node: ast.UnaryOp) -> _gir.Tensor:
        # todo modify the code below
        operand_ir = self.visit(node.operand)
        if scalar_or_int_var(operand_ir):
            op = _gir.UnaryElementWiseOperator(type(node.op))
            result = op(operand_ir)[0]
            self.kernel_p.graph_nodes.append(result)
            self.kernel_p.graph_nodes.append(op)
            return result
        else:
            raise SyntaxError(f"{type(node.op).__name__} ({operand_ir}) is not supported "
                              f"because {operand_ir} is not a scalar")

    def visit_BinOp(self, node: ast.BinOp) -> _gir.Tensor:
        lhs_ir = self.visit(node.left)
        rhs_ir = self.visit(node.right)
        if scalar_or_int_var(lhs_ir) and scalar_or_int_var(rhs_ir):
            op = _gir.BinaryElementWiseOperator(type(node.op))
            result = op(lhs_ir, rhs_ir)[0]
            self.kernel_p.graph_nodes.append(result)
            self.kernel_p.graph_nodes.append(op)
            return result
        else:
            raise SyntaxError(f"{lhs_ir} {type(node.op).__name__} {rhs_ir} is not supported "
                              f"because they are not both scalar")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        opname = type(node.op).__name__
        values = [self.visit(v) for v in node.values]
        for i in range(len(values) - 1):
            lhs = values[i]
            rhs = values[i + 1]
            if scalar_or_int_var(lhs) and scalar_or_int_var(rhs):
                op = _gir.BinaryElementWiseOperator(type(node.op))
                t = op(lhs, rhs)[0]
                self.kernel_p.graph_nodes.append(t)
                self.kernel_p.graph_nodes.append(op)
                values[i + 1] = t
            else:
                raise SyntaxError(f"{lhs} {opname} {rhs} is not supported "
                                  f"because they are not both scalar")
        return values[-1]

    def visit_Compare(self, node: ast.Compare) -> Any:
        lhs = self.visit(node.left)
        comparators = [self.visit(c) for c in node.comparators]
        for i in range(len(comparators)):
            op = node.ops[i]
            opname = type(op).__name__
            rhs = comparators[i]
            if scalar_or_int_var(lhs) and scalar_or_int_var(rhs):
                op = _gir.BinaryElementWiseOperator(type(op))
                lhs = op(lhs, rhs)[0]
                self.kernel_p.graph_nodes.append(lhs)
                self.kernel_p.graph_nodes.append(op)
            else:
                raise SyntaxError(f"{lhs} {opname} {rhs} is not supported "
                                  f"because they are not both scalar")
        return lhs

    def visit_Slice(self, node: ast.Slice) -> Any:
        """
        lower = node.lower
        upper = node.upper
        step = node.step
        if lower is None:
            lower = _ir.const(0, "int64")
        else:
            pass

        if step is None:
            step = _ir.const(1, "int64")
        else:
            step = self.visit(step)
            self.var_stack.pop()"""
        raise NotImplementedError("slice is not supported yet")

    def visit_Subscript(self, node: ast.Subscript) -> _gir.Scalar:
        # todo modify the code below
        # is load
        """
        value_ctx = self.visit(node.value)
        if isinstance(node.slice, (ast.Slice,)):
            raise SyntaxError(
                f"slicing ndarray {value_ctx.name} is not supported because it doesn't generate a scalar.")
        sls = self._get_indexing(node.slice)
        if isinstance(node.ctx, ast.Del):
            raise SyntaxError(f"del {value_ctx.name} is not allowed")
        return NDArrayIndexingNode(value_ctx, sls, self.build_span(node))"""
        raise NotImplementedError("visit_Subscript is not supported yet")

    def _get_indexing(self, sls):
        if not isinstance(sls, ast.Tuple):
            return [self.visit(sls)]
        idx = []
        for i in sls.elts:
            rt_ir = self.visit(i)
            idx.append(rt_ir)
        return idx

    def visit_Assign(self, node: ast.Assign) -> _gir.Node:
        if len(node.targets) > 1:
            raise SyntaxError(f"Assigning multiple is not allowed")
        if not isinstance(node.targets[0], (ast.Name, ast.Subscript)):
            raise SyntaxError(f"Assigning to {type(node.targets)} is not allowed.")
        value = self.visit(node.value)
        target = self.visit(node.targets[0])
        if isinstance(node.targets[0], ast.Name):
            return self._assign_scalar(target.name(), value, node)
        elif isinstance(node.targets[0], ast.Subscript):
            raise SyntaxError(f"not supported node type {type(target)}")
        else:
            raise SyntaxError(f"not supported node type {type(target)}")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        ann = self.annotation_to_kernel_type(node.annotation)
        value = self.visit(node.value)
        if not isinstance(node.target, (ast.Name, ast.Subscript)):
            raise SyntaxError(f"Assigning to {type(node.target)} is not allowed.")
        if isinstance(node.target, ast.Name):
            if typing_utils.is_scalar_type(ann):
                return self._allocate_scalar(node.target.id, value, ann)
            if typing_utils.is_ndarray_type(ann):
                return self._allocate_ndarray(node.target.id, value, ann)
        # symbol case
        elif isinstance(node.target, ast.Subscript):
            raise NotImplementedError("assigning to ndarray not supported yet")
        else:
            raise SyntaxError(f"not supported node type {type(node.target)}")

    def _allocate_scalar(self, target_name, value, ann):
        # the name is conflict with args
        if target_name in self.arg_context_table:
            raise SyntaxError(
                f"Reassigning the scalar {target_name} defined in arguments is not allowed")
        # the name is conflict with previous defined scalar
        if target_name in self.tmp_scalar_table:
            t = self.tmp_scalar_table[target_name]
            t = _gir.utils.convert_to_kernel_type(t)
            if t != ann:
                raise SyntaxError(
                    f"Reallocating the scalar {target_name} defined previous is not allowed")
        # make sure it is annotated as scalar
        if not typing_utils.is_scalar_type(ann):
            raise SyntaxError(f"Annotating {target_name} with type {ann} is not allowed.")
        # make sure the annotated type is the same as rhs value
        v_kernel = _gir.utils.convert_to_kernel_type(value)
        if v_kernel != ann:
            if v_kernel.shape != ann.shape:
                raise SyntaxError(
                    f"Assigning {value.kernel_type} to {ann} is not allowed because they have different shapes")
            elif v_kernel.dtype != ann.dtype:
                pass
            else:
                raise SyntaxError(f"Assigning {value.kernel_type} to {ann} is not allowed")
        tmp_scalar_ctx = _gir.Scalar(
            name=target_name,
            dtype=ann.dtype_str(),
            is_internal_constant=True)
        self.tmp_scalar_table[target_name] = tmp_scalar_ctx
        copy_op = _gir.CopyOperator()
        rt = copy_op(tmp_scalar_ctx, value)[0]
        self.kernel_p.graph_nodes.append(copy_op)
        self.kernel_p.graph_nodes.append(tmp_scalar_ctx)
        return rt

    def _allocate_ndarray(self, target_name, value, ann):
        # the name is conflict with args
        if target_name in self.arg_context_table:
            raise SyntaxError(
                f"Reassigning the ndarray {target_name} defined in arguments is not allowed")
        # the name is conflict with previous defined ndarray
        if target_name in self.tmp_ndarray_table and \
                _gir.utils.convert_to_kernel_type(self.tmp_ndarray_table[target_name]) != ann:
            raise SyntaxError(
                f"Reallocating the ndarray {target_name} defined previous is not allowed")
        # make sure it is annotated as scalar
        if not typing_utils.is_ndarray_type(ann):
            raise SyntaxError(f"Annotating {target_name} with type {ann} is not allowed.")
        # make sure the annotated type is the same as rhs value
        v_kernel = _gir.utils.convert_to_kernel_type(value)
        if v_kernel != ann:
            raise SyntaxError(f"Assigning {v_kernel} to {ann} is not allowed")
        # todo shape marked here may be by scalars.
        tmp_ndarray_ctx = _gir.Tensor(name=target_name, dtype=value.dtype(), shape=value.shape())
        self.tmp_ndarray_table[target_name] = tmp_ndarray_ctx
        copy_op = _gir.CopyOperator()
        rt = copy_op(tmp_ndarray_ctx, value)[0]
        self.kernel_p.graph_nodes.append(copy_op)
        self.kernel_p.graph_nodes.append(tmp_ndarray_ctx)
        return rt

    def _assign_scalar(self, target_name, value, node):
        # the name is conflict with args
        if target_name in self.arg_context_table:
            raise SyntaxError(
                f"Reassigning scalars {target_name} defined in arguments is not allowed")
        # it has not been defined
        if target_name not in self.tmp_scalar_table:
            raise SyntaxError(
                f"Assigning scalars {target_name} is not allowed because it not defined")
        # node cannot be annotated assign or other (unlikely to be other)
        if not isinstance(node, ast.Assign):
            raise SyntaxError(f"Using annotated assign to assign {target_name} is not allowed "
                              f"since it has already been defined above")
        previous_ctx = self.tmp_scalar_table[target_name]
        if _gir.utils.is_compatible(previous_ctx, value):
            raise SyntaxError(f"the value assigned to {target_name} is not scalar")
        new_ctx = _gir.Scalar(
            name=target_name,
            dtype=previous_ctx.dtype(),
            is_internal_constant=True)
        self.tmp_scalar_table[target_name] = new_ctx
        copy_op = _gir.CopyOperator()
        rt = copy_op(new_ctx, value)
        return rt[0]

    def _assign_ndarray(self, target_name, value, node):
        """
        # the name is conflict with args
        if target_name in self.arg_context_table:
            raise SyntaxError(
                f"Reassigning ndarray {target_name} defined in arguments is not allowed")
        # it has not been defined
        if target_name not in self.tmp_ndarray_table:
            raise SyntaxError(
                f"Assigning ndarray {target_name} is not allowed because it not defined")
        # node cannot be annotated assign or other (unlikely to be other)
        if not isinstance(node, ast.Assign):
            raise SyntaxError(f"Using annotated assign to assign {target_name} is not allowed "
                              f"since it has already been defined above")
        previous_ctx = self.tmp_ndarray_table[target_name]
        self.tmp_ndarray_table[target_name] = value
        if _gir.utils.is_compatible(previous_ctx, value):
            raise SyntaxError(f"the value assigned to {target_name} is not scalar")
        return value"""
        pass

    def visit_If(self, node: ast.If) -> Any:
        """
        test = self.visit(node.test)
        body = [self.visit(s) for s in node.body]
        orelse = [self.visit(s) for s in node.orelse]
        return IfNode(test, body, orelse, self.build_span(node))"""
        raise NotImplementedError("visit_If is not supported yet")

    def visit_Pass(self, node: ast.Pass) -> Any:
        lhs = _gir.Scalar(value=1, is_internal_constant=True)
        rhs = _gir.Scalar(value=2, is_internal_constant=True)
        self.kernel_p.graph_nodes.append(lhs)
        self.kernel_p.graph_nodes.append(rhs)
        op = _gir.BinaryElementWiseOperator(ast.Add)
        self.kernel_p.graph_nodes.append(op)
        result = op(lhs, rhs)[0]
        return result

    def visit_Call(self, node: ast.Call) -> Any:
        func = node.func
        if isinstance(func, ast.Name):
            f_obj = self.kernel_p.root_node.module.globals.get(func.id)
            if func.id not in FUNC_REGISTRY:
                p = matx.kernel.kernel_parser.KernelParser(f_obj)
                p.parse()
            func_inspector = FUNC_REGISTRY[id(f_obj)]
        else:
            raise SyntaxError("only support inline function for now")
        self.can_inline = self.can_inline and func_inspector.can_inline
        if func_inspector.can_inline:
            return self._inline_func(node, func_inspector)
        else:
            raise NotImplementedError("only support inline function for now")

    def _inline_func(self, node: ast.Call, inspector: 'KernelInspector'):
        tensors_nodes = (s for s in inspector.graph_input if isinstance(s, _gir.Tensor))
        for arg, tensor_node in zip(node.args, tensors_nodes):
            arg_value_node = self.visit(arg)
            cp_op = _gir.CopyOperator()
            cp_op(tensor_node, arg_value_node)
            self.kernel_p.graph_nodes.append(cp_op)
            for a_symbol, tensor_symbol in zip(arg_value_node.shape(), tensor_node.shape()):
                if isinstance(tensor_symbol, _gir.IntVar) and (not isinstance(tensor_symbol, _gir.IntImm)):
                    tensor_symbol._attrs["symbolic_value"] = a_symbol.symbolic_value()
                    tensor_symbol._attrs["name"] = a_symbol._attrs["name"]
        self.kernel_p.graph_nodes.extend(inspector.graph_nodes)
        output = inspector.graph_output[0]
        return output

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        attr_name = node.attr
        if not isinstance(value, list):
            return [value, attr_name]
        return [*value, attr_name]

    def visit_Return(self, node: ast.Return) -> Union[None, _gir.Node]:
        if node.value is None:
            return None
        if not _gir.utils.is_graph_ir_scalar(self.return_ctx):
            raise NotImplementedError(
                "base parser does not support returning things other than scalar")

        rt_ir = self.visit(node.value)
        if not _gir.utils.is_graph_ir_scalar(rt_ir):
            raise NotImplementedError(
                "The return value is not a scalar which does not match the annotation")
        copy_operator = _gir.CopyOperator()
        return copy_operator(self.return_ctx, rt_ir)[0]

    def visit_Tuple(self, node: ast.Tuple) -> List[_gir.Node]:
        values = []
        for e in node.elts:
            if not isinstance(e, ast.Name):
                raise SyntaxError(f"for now tuple only support symbol")
            if e.id not in self.kernel_p.shape_symbol_table:
                raise SyntaxError(f"for now tuple only support symbol")
            s = self.kernel_p.shape_symbol_table[e.id]
            values.append(s.symbolic_value())
        return values

    def annotation_to_kernel_type(self, ann):
        if isinstance(ann, ast.Subscript):
            if not isinstance(ann.value, ast.Name):
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get ann.value is {type(ann.value)}")
            type_name = ann.value.id
            if type_name not in typing_utils.STR_TO_KERNEL_TYPE:
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get {type_name}")
            kernel_t = typing_utils.STR_TO_KERNEL_TYPE[type_name]
            if not isinstance(ann.slice, ast.Tuple):
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get ann.slice is {type(ann.slice)}")
            return kernel_t[self.visit(ann.slice)]
        if isinstance(ann, ast.Name):
            type_name = ann.id
            if type_name not in typing_utils.STR_TO_KERNEL_TYPE:
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get {type_name}")
            return typing_utils.STR_TO_KERNEL_TYPE[type_name]
        raise SyntaxError(
            f"kernel variable can only be marked with kernel type, but get {type(ann)}")
