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
from matx._typed_ast import ast
import logging
import copy

logger = logging.getLogger('matx.py.normalize_name')


cpp_keywords = [
    'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel', 'atomic_commit', 'atomic_noexcept', 'auto',
    'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 'char8_t', 'char16_t', 'char32_t', 'class',
    'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit', 'const_cast', 'continue', 'co_await',
    'co_return', 'co_yield', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
    'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long',
    'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private',
    'protected', 'public', 'reflexpr', 'register', 'reinterpret_cast', 'requires', 'return', 'short', 'signed',
    'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 'switch', 'synchronized', 'template', 'this',
    'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
    'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq']


class NameNormalizer(ast.NodeTransformer):

    def __init__(self) -> None:
        self.change = False
        self.node_stack = []
        self.field_stack = []
        self.class_name = ''
        self.function_name = ''

    def run_impl(self, node: context.ASTNode):
        self.change = False
        # TODO: fix !532
        # last_node = copy.deepcopy(node)
        new_ast = self.visit(node.ast)
        if self.change:
            # node.last = last_node
            node.ast = new_ast

    def run(self, sc_ctx: context.ScriptContext):
        self.run_impl(sc_ctx.main_node)
        for dep_node in sc_ctx.deps_node:
            self.run_impl(dep_node)

    def rename_cpp_keyword(self, name):
        if name in cpp_keywords:
            self.change = True
            new_name = 'CPP_KW_' + name
            logger.debug(
                '[{}:{}] Rename "{}" to "{}"'.format(
                    self.class_name,
                    self.function_name,
                    name,
                    new_name))
            return new_name
        return name

    def generic_visit(self, node: ast.AST):
        self.node_stack.append(node)
        for field, old_value in ast.iter_fields(node):
            self.field_stack.append(field)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
            self.field_stack.pop(-1)
        self.node_stack.pop(-1)
        return node

    def parse_slots(self, node: ast.ClassDef) -> ast.ClassDef:
        # find __slot__
        for sub_node in node.body:
            if (isinstance(sub_node, ast.AnnAssign) and isinstance(sub_node.target, ast.Name) and
                sub_node.target.id == '__slots__') or \
                (isinstance(sub_node, ast.Assign) and isinstance(sub_node.targets[0], ast.Name) and
                 sub_node.targets[0].id == '__slots__'):
                assert isinstance(sub_node.value, ast.List)
                for i in range(len(sub_node.value.elts)):
                    assert isinstance(sub_node.value.elts[i], ast.Str)
                    sub_node.value.elts[i].s = self.rename_cpp_keyword(sub_node.value.elts[i].s)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_name = node.name
        node = self.parse_slots(node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_name = node.name
        for arg in node.args.args:
            arg.arg = self.rename_cpp_keyword(arg.arg)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(self.node_stack[-1], ast.Call) and self.field_stack[-1] == 'func':
            pass
        elif 'returns' in self.field_stack or 'annotation' in self.field_stack:
            pass
        else:
            node.raw_id = node.id
            node.id = self.rename_cpp_keyword(node.id)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(self.node_stack[-1], ast.Call) and self.field_stack[-1] == 'func':
            pass
        else:
            node.attr = self.rename_cpp_keyword(node.attr)
        return self.generic_visit(node)
