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

import abc
import types
from typing import List
import builtins
import inspect
import sys
from matx._typed_ast import ast

from .. import context
from ._python_builtin_module import is_builtin_module


class DepsAnalysis(ast.NodeVisitor):
    MATX_MODULE = sys.modules['matx']
    SKIP_MODULES = {MATX_MODULE}
    SKIP_OBJECTS = [None, False, True, open]
    if 'torch' in sys.modules:
        SKIP_MODULES.add(sys.modules['torch'])
    if 'numpy' in sys.modules:
        SKIP_MODULES.add(sys.modules['numpy'])

    def __init__(self) -> None:
        self.current_globals = None
        self.sc_ctx = None
        self.dependencies = []
        self.attr_stack = []
        self.node_stack = []
        self.field_stack = []
        self.skip_names = set()

    def run_impl(self, node: context.ASTNode,
                 prev_nodes: List[context.ASTNode]) -> List[context.ASTNode]:
        if node.deps is None:
            node.deps = []
            new_deps_ever = []

            self.current_globals = node.module.globals
            self.dependencies.clear()
            self.visit(node.ast)

            for dep, dep_node in self.dependencies:
                prev_node = self.find_node_by_raw(dep, prev_nodes)
                if prev_node is not None:
                    ast_node = prev_node
                else:
                    ast_node = context.ASTNode()
                    ast_node.raw = dep
                    self.sc_ctx.deps_relation[ast_node] = (node, dep_node)
                    new_deps_ever.append(ast_node)
                node.deps.append(ast_node)

            return new_deps_ever

        return []

    @classmethod
    def find_node_by_raw(cls, raw, nodes):
        for node in nodes:
            if node.raw is raw:
                return node
        return None

    def run(self, sc_ctx: context.ScriptContext):
        self.sc_ctx = sc_ctx

        new_deps = []
        new_deps.extend(self.run_impl(sc_ctx.main_node,
                                      sc_ctx.deps_node + [sc_ctx.main_node]))
        for dep_node in sc_ctx.deps_node:
            new_deps.extend(self.run_impl(dep_node,
                                          sc_ctx.deps_node + [sc_ctx.main_node] + new_deps))

        sc_ctx.deps_node = sc_ctx.main_node.topodeps()
        sc_ctx.deps_node = [dep for dep in sc_ctx.deps_node if dep is not sc_ctx.main_node]

        return len(new_deps) > 0

    def try_to_add_dependency(self, dep, dep_node) -> bool:
        def get_root_module(dep_cls):
            mod = inspect.getmodule(dep_cls)
            if mod is None:
                return builtins
            name = mod.__name__
            mods = name.split('.')
            if mods:
                return sys.modules[mods[0]]
            else:
                return sys.modules['__main__']

        def belong_to_module(dep_cls, belong_to):
            mod = inspect.getmodule(dep_cls)
            if mod is None:
                return False
            mod_name = mod.__name__
            mods = mod_name.split('.')
            dep_root_mod = sys.modules[mods[0]]
            for i in range(1, len(mods)):
                name = mods[i]
                dep_root_mod = getattr(dep_root_mod, name)
                if dep_root_mod is belong_to:
                    return True
            return False

        if isinstance(dep, (str, int, float, bool)):
            return False
        if isinstance(dep, types.BuiltinFunctionType):
            return False
        if hasattr(dep, "__RAW_TYPE_2_71828182846___"):
            raw_type = dep.__RAW_TYPE_2_71828182846___
            if not any(raw_type == _dep for _dep, _ in self.dependencies):
                self.dependencies.append((raw_type, dep_node))
        root_mod = get_root_module(dep)
        if is_builtin_module(root_mod):
            return False
        if (root_mod in self.SKIP_MODULES
                and not belong_to_module(dep, self.MATX_MODULE.text)
                and not belong_to_module(dep, self.MATX_MODULE.vision)
                and not belong_to_module(dep, self.MATX_MODULE.tools)):
            return False
        if dep in self.SKIP_OBJECTS:
            return False
        if not isinstance(dep, (type, types.FunctionType)):
            return False
        if getattr(dep, "__MATX_NATIVE_FUNCTION__", None):
            return False
        if getattr(dep, "__MATX_NATIVE_OBJECT__", None):
            return False
        if getattr(dep, "__MATX_NATIVE_OP__", None):
            return False
        if dep is abc.ABC:
            return False
        if not any(dep == _dep for _dep, _ in self.dependencies):
            self.dependencies.append((dep, dep_node))
        return True

    def generic_visit(self, node: ast.AST) -> None:
        self.node_stack.append(node)
        for field, value in ast.iter_fields(node):
            self.field_stack.append(field)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
            self.field_stack.pop(-1)
        self.node_stack.pop(-1)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.skip_names.update(arg.arg for arg in node.args.args)
        self.generic_visit(node)
        self.skip_names.clear()

    def visit_Name(self, node: ast.Name):
        # TODO: generate skip_names by data flow analysis
        if node.id in self.skip_names:
            return
        if self.field_stack[-1] == 'decorator_list':
            return
        if node.id in self.current_globals:
            obj = self.current_globals[node.id]
            if self.attr_stack:
                if inspect.ismodule(obj):
                    # module.Class() or module.func()
                    for attr in reversed(self.attr_stack):
                        obj = getattr(obj, attr, None)
                        if inspect.ismodule(obj):
                            # nested mod1.mod2.
                            continue
                        else:
                            break
                    if obj is not None:
                        self.try_to_add_dependency(obj, node)
                elif inspect.isclass(obj):
                    # Class().xx
                    self.try_to_add_dependency(obj, node)
                elif inspect.isfunction(obj):
                    # function.__name__
                    self.try_to_add_dependency(obj, node)
                else:
                    # object.func()
                    pass
            else:
                # Class() or func()
                self.try_to_add_dependency(obj, node)
        else:
            # unbounded name, maybe a variable
            pass

    def visit_Attribute(self, node: ast.Attribute):
        self.attr_stack.append(node.attr)
        super().generic_visit(node)
        self.attr_stack.pop()
