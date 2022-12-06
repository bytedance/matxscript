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

# Rename according to https://docs.python.org/3.7/tutorial/classes.html#private-variables
#
# > Any identifier of the form __spam (at least two leading underscores, at most
# > one trailing underscore) is textually replaced with _classname__spam, where
# > classname is the current class name with leading underscore(s) stripped. This
# > mangling is done without regard to the syntactic position of the identifier,
# > as long as it occurs within the definition of a class.
#

from .. import context
from typing import List
from matx._typed_ast import ast
import logging
import copy

logger = logging.getLogger('matx.py.rename_attrs')


class RenameAttrsTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.class_stack: List[ast.ClassDef] = []

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

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node)
        ret = self.generic_visit(node)
        self.class_stack.pop(-1)
        return ret

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name.startswith('__') and not node.name.endswith('__'):
            if self.class_stack:
                last_class = self.class_stack[-1]
                node.name = '_{}{}'.format(last_class.name, node.name)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        name = node.id
        if name.startswith('__') and not name.endswith('__'):
            if self.class_stack:
                last_class = self.class_stack[-1]
                node.id = '_{}{}'.format(last_class.name, name)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        attr = node.attr
        if attr.startswith('__') and not attr.endswith('__'):
            if self.class_stack:
                last_class = self.class_stack[-1]
                node.attr = '_{}{}'.format(last_class.name, attr)
        return self.generic_visit(node)
