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

from .. import context
from ..reporter import raise_syntax_error
from typing import List, Optional, Tuple
from matx._typed_ast import ast
import copy


class RenameCallSuperTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.class_stack: List[ast.ClassDef] = []
        self.func_stack: List[ast.FunctionDef] = []
        self.node: Optional[context.ASTNode] = None
        self.expr_to_pass = False

    def run_impl(self, node: context.ASTNode):
        self.change = False
        # TODO: fix !532
        # last_node = copy.deepcopy(node)
        self.node = node
        new_ast = self.visit(node.ast)
        if self.change:
            # node.last = last_node
            node.ast = new_ast
        self.node = None

    def run(self, sc_ctx: context.ScriptContext):
        self.run_impl(sc_ctx.main_node)
        for dep_node in sc_ctx.deps_node:
            self.run_impl(dep_node)

    def visit_Expr(self, node: ast.Expr):
        ret = self.generic_visit(node)
        if self.expr_to_pass:
            self.expr_to_pass = False
            ret = ast.Pass()
            ast.copy_location(ret, node)
        return ret

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node)
        ret = self.generic_visit(node)
        self.class_stack.pop(-1)
        return ret

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.func_stack.append(node)
        ret = self.generic_visit(node)
        self.func_stack.pop(-1)
        return ret

    def visit_Attribute(self, node: ast.Attribute):
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'super'):
            # super().xx -> self.Base::xx
            new_value, base_ctx = self.resolve_super(node)
            if base_ctx is None:
                if node.attr == '__init__':
                    # base is object, replace super().__init__ with pass
                    self.expr_to_pass = True
                    return
                raise_syntax_error(
                    self.node,
                    node,
                    "'super' object has no attribute '{}'".format(node.attr))
            node.value = new_value
            node.attr = '{}::{}'.format(base_ctx.cls_name, node.attr)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == 'super':
            raise_syntax_error(
                self.node,
                node,
                "super() with attribute is supported only: super().func() or super.x")
        ret = self.generic_visit(node)
        return ret

    def resolve_super(self, node: ast.Attribute) -> Tuple[str, context.ClassContext]:
        """resole super() and super(ty, obj) to obj.Base::x

        Returns:
            Tuple[str, str]: (obj, context of Base)
        """
        assert isinstance(node.value, ast.Call)
        argc = len(node.value.args)
        if argc == 0:
            if not self.class_stack:
                raise SyntaxError("super(): no arguments")
            cur_node = self.node.get_dep_cls_by_name(self.class_stack[-1].name, deep=True)

            obj = ast.Name()
            obj.ctx = node.ctx
            obj.id = self.func_stack[-1].args.args[0].arg
            ast.copy_location(obj, node.value)
        elif argc == 1:
            raise_syntax_error(
                self.node,
                node,
                "super with one argument is not supported now")
        elif argc == 2:
            args = node.value.args
            # TODO: inherited from module.class
            assert isinstance(args[0], ast.Name), "inherited from module.class is not supported now"
            cur_node = self.node.get_dep_cls_by_name(args[0].id, deep=True)
            obj = args[1]
        else:
            raise_syntax_error(
                self.node,
                node,
                "super() takes at most 2 arguments ({} given)".format(argc))

        if cur_node is None:
            raise_syntax_error(
                self.node,
                node,
                "super(type, obj): obj must be an instance or subtype of type")
        base_ctx = cur_node.context.base_ctx
        while base_ctx is not None:
            if node.attr == '__init__' or node.attr in base_ctx.attr_types or node.attr in base_ctx.methods:
                break
            base_ctx = base_ctx.base_ctx
        return obj, base_ctx
