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

from typing import Any, List, Dict, TYPE_CHECKING

from matx import ir as _ir
from matx.script import context as script_context
from .utils import build_span, annotation_to_kernel_type
from ..ir import *

if TYPE_CHECKING:
    from ..kernel_parser import KernelParser


class BaseParser(ast.NodeVisitor):

    def __init__(
            self,
            kernel_p: 'KernelParser',
            ndarray_context_table: Dict[str, ExpressionBaseNode],
            shape_symbol_table: Dict[str, SymbolNode],
            return_ctx: ExpressionBaseNode,
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
                if isinstance(res, StatementBaseNode):
                    res = res.to_matx_ir()
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
            if not (isinstance(ctx, NDArrayNode) or isinstance(ctx, ScalarNode)):
                continue
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
            raise SyntaxError("None is not allowed")
        elif isinstance(node.value, numbers.Number):
            dtype = get_dtype(node.value)

            const_scalar_ctx = ConstScalarNode(node.value,
                                               PYTYPE_TO_KERNEL_TYPE[dtype],
                                               build_span(self.root_node, node))
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
        if name in self.ndarray_context_table:
            ctx = self.ndarray_context_table[name]
            return ctx
        if name in self.tmp_scalar_table:
            ctx = self.tmp_scalar_table[name]
            return ctx
        raise NotImplementedError(f'the type of {name} is not support')
        # return node.id

    # Expressions
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        raise NotImplementedError("visit_UnaryOp is not Implemented")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        opname = type(node.op).__name__
        lhs_ir = self.visit(node.left)
        rhs_ir = self.visit(node.right)
        if (is_scalar_type(lhs_ir.kernel_type) or is_symbol_type(lhs_ir.kernel_type))\
                and (is_scalar_type(rhs_ir.kernel_type) or is_symbol_type(rhs_ir.kernel_type)):
            return ArithmeticBinaryOp(lhs_ir, rhs_ir, type(node.op), self.build_span(node))
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
        ann = annotation_to_kernel_type(node.annotation)
        value = self.visit(node.value)
        if not isinstance(node.target, (ast.Name, ast.Subscript)):
            raise SyntaxError(f"Assigning to {type(node.target)} is not allowed.")
        if isinstance(node.target, ast.Name):
            return self._allocate_scalar(node.target.id, value, ann, span)
        # symbol case
        elif isinstance(node.target, ast.Subscript):
            raise NotImplementedError("assigning to ndarray not supported yet")
        else:
            raise SyntaxError(f"not supported node type {type(node.target)}")

    def _allocate_scalar(self, target_name, value, ann, span):
        # the name is conflict with args
        if target_name in self.ndarray_context_table:
            raise SyntaxError(
                f"Reassigning scalars {target_name} defined in arguments is not allowed")
        # the name is conflict with previous defined scalar
        if target_name in self.tmp_scalar_table:
            raise SyntaxError(f"Reallocating scalars {target_name} defined previous is not allowed")
        # make sure it is annotated as scalar
        if not is_scalar_type(ann):
            raise SyntaxError(f"Annotating {target_name} with type {ann} is not allowed.")
        # make sure the annotated type is the same as rhs value
        if value.kernel_type != ann:
            raise SyntaxError(f"Assigning {value.kernel_type} to {ann} is not allowed")
        tmp_scalar_ctx = ScalarNode(target_name, ann, span)
        self.tmp_scalar_table[target_name] = tmp_scalar_ctx
        return AllocationScalarNode(tmp_scalar_ctx, value, span)

    def _assign_scalar(self, target_name, value, node, span):
        # the name is conflict with args
        if target_name in self.ndarray_context_table:
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

    def _assign_ndarray(self):
        raise SyntaxError("Assigning to ndarray is not allowed")

    def visit_Pass(self, node: ast.Pass) -> Any:
        lhs = ConstScalarNode(1, kernelNDArrayT((1, 1), np.int32), self.build_span(node))
        rhs = ConstScalarNode(2, kernelNDArrayT((1, 1), np.int32), self.build_span(node))
        return ArithmeticBinaryOp(lhs, rhs, ast.Add, self.build_span(node))

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
