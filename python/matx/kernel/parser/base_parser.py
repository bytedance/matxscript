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
from typing import Any, TYPE_CHECKING

from matx import ir as _ir
from ..ir import *

if TYPE_CHECKING:
    from ..kernel_parser import KernelInspector


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

    def visit_Constant(self, node: ast.Constant) -> Any:
        if node.value is None:
            raise SyntaxError("None is not allowed")
        elif isinstance(node.value, numbers.Number):
            dtype = get_dtype(node.value)

            const_scalar_ctx = ConstScalarNode(node.value,
                                               PYTYPE_TO_KERNEL_TYPE[dtype],
                                               None)
            return const_scalar_ctx
        else:
            raise NotImplementedError(f'Unsupported value {node.value}')

    # variables
    def visit_Name(self, node: ast.Name) -> Any:
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
        raise NotImplementedError(f'the type of {name} is not support')
        # return node.id

    # Expressions
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        opname = type(node.op).__name__
        operand_ir = self.visit(node.operand)
        if (is_scalar_type(operand_ir.kernel_type) or is_symbol_type(operand_ir.kernel_type)):
            return UnaryOp(operand_ir, type(node.op), self.build_span(node))
        else:
            raise SyntaxError(f"{opname} ({operand_ir}) is not supported "
                              f"because {operand_ir} is not a scalar")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        opname = type(node.op).__name__
        lhs_ir = self.visit(node.left)
        rhs_ir = self.visit(node.right)
        if (is_scalar_type(lhs_ir.kernel_type) or is_symbol_type(lhs_ir.kernel_type)) \
                and (is_scalar_type(rhs_ir.kernel_type) or is_symbol_type(rhs_ir.kernel_type)):
            return BinaryOp(lhs_ir, rhs_ir, type(node.op), self.build_span(node))
        else:
            raise SyntaxError(f"{lhs_ir} {opname} {rhs_ir} is not supported "
                              f"because they are not both scalar")

    def visit_Slice(self, node: ast.Slice) -> Any:
        raise NotImplementedError("slice is not supported yet")
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
            self.var_stack.pop()

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # is load
        value_ctx = self.visit(node.value)
        if isinstance(node.slice, (ast.Slice,)):
            raise SyntaxError(
                f"slicing ndarray {value_ctx.name} is not supported because it doesn't generate a scalar.")
        sls = self._get_indexing(node.slice)
        if isinstance(node.ctx, ast.Del):
            raise SyntaxError(f"del {value_ctx.name} is not allowed")
        return NDArrayIndexingNode(value_ctx, sls, self.build_span(node))

    def _get_indexing(self, sls):
        if not isinstance(sls, ast.Tuple):
            return [self.visit(sls)]
        idx = []
        for i in sls.elts:
            rt_ir = self.visit(i)
            idx.append(rt_ir)
        return idx

    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) > 1:
            raise SyntaxError(f"Assigning multiple is not allowed")
        if not isinstance(node.targets[0], (ast.Name, ast.Subscript)):
            raise SyntaxError(f"Assigning to {type(node.targets)} is not allowed.")
        span = self.build_span(node)
        value = self.visit(node.value)
        target = self.visit(node.targets[0])
        if isinstance(node.targets[0], ast.Name):
            return self._assign_scalar(target.name, value, node, span)
        elif isinstance(node.targets[0], ast.Subscript):
            return AssignScalarNode(target, value, span)
        else:
            raise SyntaxError(f"not supported node type {type(node.target)}")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        span = self.build_span(node)
        ann = self.annotation_to_kernel_type(node.annotation)
        value = self.visit(node.value)
        if not isinstance(node.target, (ast.Name, ast.Subscript)):
            raise SyntaxError(f"Assigning to {type(node.target)} is not allowed.")
        if isinstance(node.target, ast.Name):
            if is_scalar_type(ann):
                return self._allocate_scalar(node.target.id, value, ann, span)
            if is_ndarray_type(ann):
                return self._allocate_ndarray(node.target.id, value, ann, span)
        # symbol case
        elif isinstance(node.target, ast.Subscript):
            raise NotImplementedError("assigning to ndarray not supported yet")
        else:
            raise SyntaxError(f"not supported node type {type(node.target)}")

    def _allocate_scalar(self, target_name, value, ann, span):
        # the name is conflict with args
        if target_name in self.arg_context_table:
            raise SyntaxError(
                f"Reassigning the scalar {target_name} defined in arguments is not allowed")
        # the name is conflict with previous defined scalar
        if target_name in self.tmp_scalar_table and self.tmp_scalar_table[target_name].kernel_type != ann:
            raise SyntaxError(
                f"Reallocating the scalar {target_name} defined previous is not allowed")
        # make sure it is annotated as scalar
        if not is_scalar_type(ann):
            raise SyntaxError(f"Annotating {target_name} with type {ann} is not allowed.")
        # make sure the annotated type is the same as rhs value
        if value.kernel_type != ann:
            if value.kernel_type.shape != ann.shape:
                raise SyntaxError(
                    f"Assigning {value.kernel_type} to {ann} is not allowed because they have different shapes")
            elif value.kernel_type.dtype != ann.dtype:
                value = CastOp(value, ann.dtype, span)
            else:
                raise SyntaxError(f"Assigning {value.kernel_type} to {ann} is not allowed")
        tmp_scalar_ctx = ScalarNode(target_name, ann, span)
        self.tmp_scalar_table[target_name] = tmp_scalar_ctx
        return ScalarAllocationNode(tmp_scalar_ctx, value, span)

    def _allocate_ndarray(self, target_name, value, ann, span):
        # the name is conflict with args
        if target_name in self.arg_context_table:
            raise SyntaxError(
                f"Reassigning the ndarray {target_name} defined in arguments is not allowed")
        # the name is conflict with previous defined ndarray
        if target_name in self.tmp_ndarray_table and self.tmp_ndarray_table[target_name].kernel_type != ann:
            raise SyntaxError(
                f"Reallocating the ndarray {target_name} defined previous is not allowed")
        # make sure it is annotated as scalar
        if not is_ndarray_type(ann):
            raise SyntaxError(f"Annotating {target_name} with type {ann} is not allowed.")
        # make sure the annotated type is the same as rhs value
        if value.kernel_type != ann:
            raise SyntaxError(f"Assigning {value.kernel_type} to {ann} is not allowed")
        # todo shape marked here may be by scalars.
        tmp_ndarray_ctx = NDArrayNode(target_name, ann, self.shape_symbol_table, span)
        self.tmp_ndarray_table[target_name] = tmp_ndarray_ctx
        return ScopedNDArrayAllocationNode(
            tmp_ndarray_ctx, value, self.kernel_p.continue_parse_as_scoped(), span)

    def _assign_scalar(self, target_name, value, node, span):
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
        if value.kernel_type != previous_ctx.kernel_type:
            raise SyntaxError(f"the value assigned to {target_name} is not scalar")
        return AssignScalarNode(previous_ctx, value, span)

    def _assign_ndarray(self, target_name, value, node, span):
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
        if value.kernel_type != previous_ctx.kernel_type:
            raise SyntaxError(f"the value assigned to {target_name} is not scalar")
        return AssignNDArrayNode(previous_ctx, value)

    def visit_If(self, node: ast.If) -> Any:
        test = self.visit(node.test)
        body = [self.visit(s) for s in node.body]
        orelse = [self.visit(s) for s in node.orelse]
        return IfNode(test, body, orelse, self.build_span(node))

    def visit_Pass(self, node: ast.Pass) -> Any:
        lhs = ConstScalarNode(1, kernelNDArrayT((1, 1), np.int32), self.build_span(node))
        rhs = ConstScalarNode(2, kernelNDArrayT((1, 1), np.int32), self.build_span(node))
        return BinaryOp(lhs, rhs, ast.Add, self.build_span(node))

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        opname = type(node.op).__name__
        values = [self.visit(v) for v in node.values]
        for i in range(len(values) - 1):
            lhs = values[i]
            rhs = values[i + 1]
            if (is_scalar_type(lhs.kernel_type) or is_symbol_type(lhs.kernel_type)) \
                    and (is_scalar_type(rhs.kernel_type) or is_symbol_type(rhs.kernel_type)):
                t = BinaryOp(lhs, rhs, type(node.op), self.build_span(node))
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
            if (is_scalar_type(lhs.kernel_type) or is_symbol_type(lhs.kernel_type)) \
                    and (is_scalar_type(rhs.kernel_type) or is_symbol_type(rhs.kernel_type)):
                lhs = BinaryOp(lhs, rhs, type(op), self.build_span(node))
            else:
                raise SyntaxError(f"{lhs} {opname} {rhs} is not supported "
                                  f"because they are not both scalar")
        return lhs

    def visit_Call(self, node: ast.Call) -> Any:
        raise NotImplementedError("visit_Call is not Implemented")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        attr_name = node.attr
        if not isinstance(value, list):
            return [value, attr_name]
        return [*value, attr_name]

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is None:
            return _ir.ReturnStmt(NoneExpr())
        if not is_scalar_type(self.return_ctx.kernel_type):
            raise NotImplementedError(
                "base parser does not support returning things other than scalar")

        rt_ir = self.visit(node.value)
        if not is_scalar_shape(rt_ir.shape):
            raise NotImplementedError(
                "The return value is not a scalar which does not match the annotation")

        return _ir.ReturnStmt(rt_ir.to_matx_ir())

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        values = []
        for e in node.elts:
            if not isinstance(e, ast.Name):
                raise SyntaxError(f"for now tuple only support symbol")
            if e.id not in self.kernel_p.shape_symbol_table:
                raise SyntaxError(f"for now tuple only support symbol")
            s = self.kernel_p.shape_symbol_table[e.id]
            values.append(s.symbol)
        return values

    def annotation_to_kernel_type(self, ann):
        if isinstance(ann, ast.Subscript):
            if not isinstance(ann.value, ast.Name):
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get ann.value is {type(ann.value)}")
            type_name = ann.value.id
            if type_name not in STR_TO_KERNEL_TYPE:
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get {type_name}")
            kernel_t = STR_TO_KERNEL_TYPE[type_name]
            if not isinstance(ann.slice, ast.Tuple):
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get ann.slice is {type(ann.slice)}")
            return kernel_t[self.visit(ann.slice)]
        if isinstance(ann, ast.Name):
            type_name = ann.id
            if type_name not in STR_TO_KERNEL_TYPE:
                raise SyntaxError(
                    f"kernel variable can only be marked with kernel type, but get {type_name}")
            return STR_TO_KERNEL_TYPE[type_name]
        raise SyntaxError(
            f"kernel variable can only be marked with kernel type, but get {type(ann)}")
