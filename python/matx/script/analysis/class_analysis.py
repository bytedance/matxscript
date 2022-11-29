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
from typing import Any
import inspect

from ..context.function_context import FunctionContext, FunctionType
from .. import context
from .function_analysis import FunctionAnalysis
from .function_analysis import CollectFuncArgsBindsInfo
from ..type_parser import parse_type
from ..rules import NameRule
from ..reporter import raise_syntax_error
from ... import ir as _ir


def _get_all_attrs(ctx: context.ClassContext, ignore_self: bool = False):
    attrs = set()
    if not ignore_self:
        attrs.update(ctx.attr_names)
    if ctx.base_ctx is not None:
        attrs.update(_get_all_attrs(ctx.base_ctx, ignore_self=False))
    return attrs


class ClassAnalysis(ast.NodeVisitor):

    def __init__(self) -> None:
        self.cls_ctx = None
        self.custom_node = None
        self.current_ast_node = None

    @classmethod
    def parse_types_in_context(
        cls,
        ctx: context.ClassContext,
        node: context.ASTNode,
        sc_ctx: context.ScriptContext,
    ) -> None:
        if len(ctx.attr_types):
            for attr_name in ctx.attr_names:
                # TODO: remove isinstance check
                if isinstance(ctx.attr_types[attr_name], ast.AST):
                    ctx.attr_types[attr_name] = parse_type(
                        ctx.attr_types[attr_name],
                        node,
                        sc_ctx,
                    )
        if len(ctx.annotation_slot_names):
            # TODO: remove isinstance check
            if isinstance(ctx.annotation_slot_ann, ast.AST):
                parse_type_tuple = parse_type(ctx.annotation_slot_ann, node, sc_ctx)
                for index, attr_name in enumerate(ctx.annotation_slot_names):
                    if attr_name not in ctx.attr_types:
                        ctx.attr_types[attr_name] = parse_type_tuple[index]

        # Remove duplicated members in bases
        attrs_in_bases = _get_all_attrs(ctx, True)
        ctx.attr_names = [
            attr_name for attr_name in ctx.attr_names if attr_name not in attrs_in_bases]
        ctx.annotation_slot_names = [
            attr_name for attr_name in ctx.annotation_slot_names if attr_name not in attrs_in_bases]
        for k in attrs_in_bases:
            ctx.attr_types.pop(k, None)

        if not ctx.init_fn:
            raise_syntax_error(node, node.ast, "Class for MATX should always have an __init__.")

        fnt_anls = FunctionAnalysis()
        fnt_anls.parse_types_in_context(
            ctx.init_fn, node, sc_ctx,
        )
        for method_name, method_func in ctx.methods.items():
            fnt_anls.parse_types_in_context(
                method_func, node, sc_ctx,
            )

    def run_impl(self, node: context.ASTNode, sc_ctx: context.ScriptContext, action: str):
        self.custom_node = node
        if isinstance(node.ast, ast.ClassDef):
            if action == 'INIT':
                # Parse context
                self.cls_ctx = context.ClassContext(node.ast.name)
                self.cls_ctx.raw = node.raw
                node.context = self.cls_ctx
                node.context.cls_id = id(node.raw)
                self.visit(node.ast)
                self.cls_ctx = None
                # Init ir_schema early
                if node.ast.bases and len(node.ast.bases) > 0:
                    assert len(node.ast.bases) == 1
                    # TODO: inherited from module.class
                    assert isinstance(
                        node.ast.bases[0], ast.Name), "inherited from module.class is not supported now"
                    base_node = node.get_dep_cls_by_name(node.ast.bases[0].id)
                    if base_node:
                        node.context.base_ctx = base_node.context
                        node.context.is_abc |= base_node.context.is_abc
                        node.context.abstract_methods.update(base_node.context.abstract_methods)
                        node.context.abstract_methods.difference_update(
                            func.name for func in node.context.methods.values() if not func.is_abstract)
                        node.ir_schema = node.context.init_ir_schema(base_node.ir_schema)
                    else:
                        # TODO: check why base_node is None ?
                        node.ir_schema = node.context.init_ir_schema()
                else:
                    node.ir_schema = node.context.init_ir_schema()
            elif action == 'TYPE':
                # Parse types
                self.parse_types_in_context(
                    node.context, node, sc_ctx,
                )
                for func_name, func in inspect.getmembers(node.raw, predicate=inspect.isfunction):
                    if func_name == '__init__':
                        ctx = node.context.init_fn
                    elif func_name in node.context.methods:
                        ctx = node.context.methods[func_name]
                    else:
                        # Perhaps func is defined in base classes of node.raw
                        continue
                    sig = inspect.signature(func)
                    ctx.arg_defaults = [
                        param.default for param in sig.parameters.values() if param.default is not inspect._empty]
                node.ir_schema = node.context.update_ir_schema(node.ir_schema)  # Update
                node.context.class_instance_var = _ir.HLOVar("this", node.ir_schema)
                node.context.session_pointer_var = _ir.adt.get_implicit_class_session_var()

    def visit(self, node: ast.AST) -> Any:
        last_cur_node = self.current_ast_node
        self.current_ast_node = node
        visit_res = super(ClassAnalysis, self).visit(node)
        self.current_ast_node = last_cur_node
        return visit_res

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) == 1 and isinstance(node.body[0], (ast.ClassDef, ast.FunctionDef)):
            # class or single function
            return self.visit(node.body[0])
        raise_syntax_error(self.custom_node,
                           self.current_ast_node,
                           'Only one-function, one-class source code is allowed.')

    def visit_ClassDef(self, node: ast.ClassDef):
        self.cls_ctx.cls_name = node.name  # Move to outside?
        for base in node.bases:
            if (isinstance(base, ast.Name) and base.id == 'ABC') or \
                    (isinstance(base, ast.Attribute) and base.attr == 'ABC'):
                self.cls_ctx.is_abc = True

        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and stmt.target.id == '__slots__':
                self.cls_ctx.annotation_slot_ann = stmt.annotation
                self.cls_ctx.annotation_slot_names = [n.s for n in stmt.value.elts]
                continue
            if not isinstance(stmt, ast.FunctionDef):
                continue  # Will skip empty slot def
            fn_name = stmt.name
            fn_ctx: FunctionContext = self.visit(stmt)
            fn_ctx.raw = getattr(self.cls_ctx.raw, fn_name)
            fn_ctx.fn_type = FunctionType.INSTANCE
            fn_ctx.unbound_name = NameRule.rename_class_method(
                self.cls_ctx.cls_name, fn_ctx.fn_name)
            if stmt.name == '__init__':
                self.cls_ctx.init_fn = fn_ctx
                for stmt_inbody in stmt.body:
                    self.visit(stmt_inbody)
            else:
                self.cls_ctx.methods[fn_name] = fn_ctx

    def visit_FunctionDef(self, node: ast.FunctionDef) -> FunctionContext:
        # fn = getattr(self.cls, node.name)
        all_arg_names = []
        for arg in node.args.args:
            all_arg_names.append(arg.arg)
        args_analysis = CollectFuncArgsBindsInfo(all_arg_names)
        fn_anly = FunctionAnalysis()
        fn_anly.fn_ctx.arg_reassigns = args_analysis.run(node.body)
        fn_anly.current_ast_node = node
        fn_anly.custom_node = self.custom_node
        fn_anly.visit(node)
        fn_ctx = fn_anly.fn_ctx
        fn_anly.fn_ctx = None
        if fn_ctx.is_abstract:
            self.cls_ctx.abstract_methods.add(fn_ctx.name)
        else:
            self.cls_ctx.abstract_methods.discard(fn_ctx.name)
        del fn_anly
        return fn_ctx

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """ self.x: Annotation = Expr """
        if (not isinstance(node.target, ast.Attribute)):
            return
        if (not isinstance(node.target.value, ast.Name)):
            return
        if (node.target.value.id != 'self'):
            return

        if node.target.attr in self.cls_ctx.slot_names:
            return
        attr_name = node.target.attr
        self.cls_ctx.attr_names.append(attr_name)
        self.cls_ctx.attr_types[attr_name] = node.annotation
