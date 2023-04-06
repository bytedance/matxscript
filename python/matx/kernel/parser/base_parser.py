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
from matx.ir import generic as _generic
from matx.script import context as script_context
from .context import *
from .utils import build_span, annotation_to_kernel_type

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class BaseParser(ast.NodeVisitor):
    _binop_maker = {
        ast.Add: lambda lhs, rhs, span: _generic.add(lhs, rhs, span),
        ast.Sub: lambda lhs, rhs, span: _generic.subtract(lhs, rhs, span),
        ast.Mult: lambda lhs, rhs, span: _generic.multiply(lhs, rhs, span),
        ast.Div: lambda lhs, rhs, span: _generic.divide(lhs, rhs, span),
        ast.FloorDiv: lambda lhs, rhs, span: _generic.floordiv(lhs, rhs, span),
        ast.Mod: lambda lhs, rhs, span: _generic.floormod(lhs, rhs, span),
        ast.BitOr: lambda lhs, rhs, span: _generic.bitwise_or(lhs, rhs, span),
        ast.BitAnd: lambda lhs, rhs, span: _generic.bitwise_and(lhs, rhs, span),
        ast.BitXor: lambda lhs, rhs, span: _generic.bitwise_xor(lhs, rhs, span),
        ast.LShift: lambda lhs, rhs, span: _generic.left_shift(lhs, rhs, span),
        ast.RShift: lambda lhs, rhs, span: _generic.right_shift(lhs, rhs, span),
        ast.Gt: lambda lhs, rhs, span: _generic.greater_than(lhs, rhs, span),
        ast.GtE: lambda lhs, rhs, span: _generic.greater_or_equal(lhs, rhs, span),
        ast.Lt: lambda lhs, rhs, span: _generic.less_than(lhs, rhs, span),
        ast.LtE: lambda lhs, rhs, span: _generic.less_or_equal(lhs, rhs, span),
        ast.Eq: lambda lhs, rhs, span: _generic.equal(lhs, rhs, span),
        ast.NotEq: lambda lhs, rhs, span: _generic.notequal(lhs, rhs, span),
        ast.Is: lambda lhs, rhs, span: _generic.op_is(lhs, rhs, span),
        ast.IsNot: lambda lhs, rhs, span: _generic.op_not(_generic.op_is(lhs, rhs, span), span)
    }

    _unaryop_maker = {
        ast.USub: lambda operand, span: _generic.multiply(operand, _ir.const(-1), span),
        ast.Invert: lambda operand, span: _generic.bitwise_not(operand, span),
        ast.Not: lambda operand, span: _generic.op_not(operand, span)
    }

    _boolop_marker = {
        ast.And: lambda span, *args: _generic.op_and(span, *args),
        ast.Or: lambda span, *args: _generic.op_or(span, *args)
    }

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
        self.tmp_scalar_table = {}
        self.return_ctx = return_ctx

        self.var_stack = []
        self.reads = []

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

    def build_span(self, node):
        return build_span(self.root_node, node)

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
            self.context.update_symbol(arg, ctx.script_var)
            self.context.func_params.append(ctx.script_var)

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
            self.var_stack.append(None)
            return _ir.NoneExpr()
        elif isinstance(node.value, numbers.Number):
            dtype = get_dtype(node.value)

            const_scalar_ctx = ConstScalarContext(node.value,
                                                  PYTYPE_TO_KERNEL_TYPE[dtype],
                                                  build_span(self.root_node, node))

            self.var_stack.append(const_scalar_ctx)  # todo none for now
            return const_scalar_ctx.script_var
        else:
            raise NotImplementedError(f'Unsupported value {node.value}')

    # variables
    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Store):
            self.var_stack.append(None)
            return node.id
        if not isinstance(node.ctx, ast.Load):
            raise SyntaxError(f"del {node.id} is not allowed")
        name = node.id
        if name in self.shape_symbol_table:
            ctx = self.shape_symbol_table[name]
            self.var_stack.append(ctx)
            return ctx.script_var
        if name in self.ndarray_context_table:
            ctx = self.ndarray_context_table[name]
            self.var_stack.append(ctx)
            return ctx.script_var
        if name in self.tmp_scalar_table:
            ctx = self.tmp_scalar_table[name]
            self.var_stack.append(ctx)
            return ctx.script_var
        raise NotImplementedError(f'the type of {name} is not support')
        # return node.id

    # Expressions
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        raise NotImplementedError("visit_UnaryOp is not Implemented")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        opname = type(node.op).__name__
        lhs_ir = self.visit(node.left)
        lhs_ctx = self.var_stack.pop()
        rhs_ir = self.visit(node.right)
        rhs_ctx = self.var_stack.pop()
        if is_scalar_type(lhs_ctx.kernel_type) and is_scalar_type(rhs_ctx.kernel_type):
            if lhs_ctx.kernel_type != rhs_ctx.kernel_type:
                # todo support different type
                raise SyntaxError("type is different")
            self.var_stack.append(AbstractScalarContext(lhs_ctx.kernel_type))
            return self._binop_maker[type(node.op)](lhs_ir, rhs_ir, self.build_span(node))
        else:
            raise SyntaxError(f"{lhs_ctx.name} {opname} {rhs_ctx.name} is not supported "
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
        _ = self.visit(node.value)
        value_ctx = self.var_stack.pop()
        if value_ctx.is_abstract_ctx():
            raise SyntaxError(f"only support indexing ndarray")
        if isinstance(node.slice, (ast.Slice,)):
            raise SyntaxError(
                f"slicing ndarray {value_ctx.name} is not supported because it doesn't generate a scalar.")
        sls = self._get_indexing(node.slice)
        if isinstance(node.ctx, ast.Load):
            self.var_stack.append(value_ctx.data_ctx())
            return value_ctx.read_at(sls)
        if isinstance(node.ctx, ast.Store):
            rhs = self.var_stack.pop()
            rhs = _ir.PrimCast(value_ctx.kernel_type.dtype_str(), rhs)
            rt_ir = value_ctx.write_at(sls, rhs)
            self.var_stack.append(value_ctx)
            return rt_ir
        raise SyntaxError(f"del {value_ctx.name} is not allowed")

    def _get_indexing(self, sls):
        if not isinstance(sls, ast.Tuple):
            return [self.visit(sls)]
        idx = []
        for i in sls.elts:
            rt_ir = self.visit(i)
            self.var_stack.pop()
            idx.append(rt_ir)
        return idx

    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) > 1:
            raise SyntaxError(f"Assigning multiple is not allowed")
        if not isinstance(node.targets[0], (ast.Name, ast.Subscript)):
            raise SyntaxError(f"Assigning to {type(node.targets)} is not allowed.")
        span = self.build_span(node)
        value = self.visit(node.value)
        value_ctx = self.var_stack.pop()
        self.var_stack.append(value)
        target = self.visit(node.targets[0])
        if isinstance(node.targets[0], ast.Name):
            self.var_stack.pop()
            return self._assign_scalar(target, value, value_ctx, node, span)
        elif isinstance(node.targets[0], ast.Subscript):
            return target

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        span = self.build_span(node)
        ann = annotation_to_kernel_type(node.annotation)
        value = self.visit(node.value)
        value_ctx = self.var_stack.pop()
        if not isinstance(node.target, (ast.Name, ast.Subscript)):
            raise SyntaxError(f"Assigning to {type(node.target)} is not allowed.")
        if isinstance(node.target, ast.Name):
            return self._allocate_scalar(node.target.id, value, value_ctx, ann, span)
        # symbol case
        elif isinstance(node.target, ast.Subscript):
            raise NotImplementedError("assigning to ndarray not supported yet")

    def _allocate_scalar(self, target, value, value_ctx, ann, span):
        # the name is conflict with args
        if target in self.ndarray_context_table:
            raise SyntaxError(f"Reassigning scalars {target} defined in arguments is not allowed")
        # the name is conflict with previous defined scalar
        if target in self.tmp_scalar_table:
            raise SyntaxError(f"Reallocating scalars {target} defined previous is not allowed")
        # make sure it is annotated as scalar
        if not is_scalar_type(ann):
            raise SyntaxError(f"Annotating {target} with type {ann} is not allowed.")
        # make sure the annotated type is the same as rhs value
        if value_ctx != ann:
            raise SyntaxError(f"Assigning {value_ctx.kernel_type} to {ann} is not allowed")
        tmp_scalar_ctx = ScalarContext(target, ann, span)
        self.tmp_scalar_table[target] = tmp_scalar_ctx
        alloca_stmt = _ir.AllocaVarStmt(
            tmp_scalar_ctx.name, tmp_scalar_ctx.script_type, value, span)
        self.var_stack.append(tmp_scalar_ctx)
        return alloca_stmt

    def _assign_scalar(self, target, value, value_ctx, node, span):
        # the name is conflict with args
        if target in self.ndarray_context_table:
            raise SyntaxError(f"Reassigning scalars {target} defined in arguments is not allowed")
        # it has not been defined
        if target not in self.tmp_scalar_table:
            raise SyntaxError(f"Assigning scalars {target} is not allowed because it not defined")
        # node cannot be annotated assign or other (unlikely to be other)
        if not isinstance(node, ast.Assign):
            raise SyntaxError(f"Using annotated assign to assign {target} is not allowed "
                              f"since it has already been defined above")
        previous_ctx = self.tmp_scalar_table[target]
        if value_ctx.kernel_type != previous_ctx.kernel_type:
            raise SyntaxError(f"the value assigned to {target} is not scalar")
        self.var_stack.append(value_ctx)
        return _ir.AssignStmt(previous_ctx.script_var, value, span)

    def _assign_ndarray(self):
        raise SyntaxError("Assigning to ndarray is not allowed")

    def visit_Pass(self, node: ast.Pass) -> Any:
        self.var_stack.append(AbstractBaseVariableContext(int32))
        return _ir.NoneExpr()

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
        pass
