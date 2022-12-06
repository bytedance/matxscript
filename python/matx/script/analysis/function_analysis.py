# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from matx._typed_ast import ast
import inspect
from typing import List

from ..context.function_context import FunctionContext
from ..reporter import raise_syntax_error
from .. import context
from ..type_parser import parse_type
from ...ir import type as _type


class CollectFuncArgsBindsInfo(ast.NodeVisitor):
    def __init__(self, arg_names: List[str]) -> None:
        self._arg_names = set(arg_names)
        self._arg_rebind = {x: False for x in arg_names}

    def run(self, node: List[ast.stmt]):
        for stmt in node:
            self.visit(stmt)
        return self._arg_rebind

    def visit_Name(self, node: ast.Name):
        if node.id not in self._arg_names:
            return
        if isinstance(node.ctx, ast.Store):
            self._arg_rebind[node.id] = True


class FunctionAnalysis(ast.NodeVisitor):

    def __init__(self) -> None:
        self.fn_ctx = FunctionContext()
        self.custom_node = None
        self.current_ast_node = None

    @classmethod
    def parse_types_in_context(cls,
                               ctx: FunctionContext,
                               node: context.ASTNode,
                               sc_ctx: context.ScriptContext, ) -> None:
        ctx.return_type = parse_type(
            ctx.return_type,
            node,
            sc_ctx,
        )
        for arg_name in ctx.arg_names:
            ctx.arg_types[arg_name] = parse_type(
                ctx.arg_types[arg_name],
                node,
                sc_ctx,
            )
            if not ctx.arg_reassigns[arg_name]:
                if isinstance(ctx.arg_types[arg_name], _type.StringType):
                    ctx.arg_types[arg_name] = _type.StringType(is_view=True)
                elif isinstance(ctx.arg_types[arg_name], _type.UnicodeType):
                    ctx.arg_types[arg_name] = _type.UnicodeType(is_view=True)
                elif isinstance(ctx.arg_types[arg_name], _type.ObjectType):
                    ctx.arg_types[arg_name] = _type.ObjectType(is_view=True)

    def run_impl(self, node: context.ASTNode, sc_ctx: context.ScriptContext, action: str):
        self.custom_node = node
        self.current_ast_node = node.ast

        if isinstance(node.ast, ast.FunctionDef):
            if action == 'INIT':
                all_arg_names = []
                for arg in node.ast.args.args:
                    all_arg_names.append(arg.arg)
                args_analysis = CollectFuncArgsBindsInfo(all_arg_names)
                self.fn_ctx = context.FunctionContext()
                self.fn_ctx.arg_reassigns = args_analysis.run(node.ast.body)
                self.fn_ctx.raw = node.raw
                node.context = self.fn_ctx
                self.visit(node.ast)
                self.fn_ctx = None
            elif action == 'TYPE':
                self.parse_types_in_context(
                    node.context, node, sc_ctx
                )
                sig = inspect.signature(node.raw)
                node.context.arg_defaults = [
                    param.default for param in sig.parameters.values() if
                    param.default is not inspect._empty
                ]
                node.ir_schema = node.context.to_ir_schema()

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) == 1 and isinstance(node.body[0], ast.FunctionDef):
            return self.visit(node.body[0])
        else:
            raise_syntax_error(self.custom_node,
                               self.current_ast_node,
                               'Only one-function, one-class source code is allowed.')

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.fn_ctx.fn_name = node.name
        self.fn_ctx.unbound_name = node.name

        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod') or \
                    (isinstance(decorator, ast.Attribute) and decorator.attr == 'abstractmethod'):
                self.fn_ctx.is_abstract = True

        args = list(node.args.args)
        if node.returns is None:
            if node.name == "__init__":
                node.returns = ast.Name()
                node.returns.ctx = ast.Load()
                node.returns.lineno = node.lineno
                node.returns.col_offset = node.col_offset
                node.returns.id = 'Any'
                node.returns.value = None
            else:
                raise_syntax_error(self.custom_node,
                                   self.current_ast_node,
                                   'Function should be annotated with its return type')

        args.append(ast.arg(
            arg='return value',
            annotation=node.returns,
            lineno=node.returns.lineno,
            col_offset=node.returns.col_offset
        ))

        for arg in args:
            if arg.arg == 'self':
                continue
            if arg.annotation is None:
                raise_syntax_error(self.custom_node,
                                   self.current_ast_node,
                                   f'Missing type annotation for arg {arg.arg}')

            if arg.arg == 'return value':
                self.fn_ctx.return_type = arg.annotation
            else:
                self.fn_ctx.arg_names.append(arg.arg)
                self.fn_ctx.arg_types[arg.arg] = arg.annotation
