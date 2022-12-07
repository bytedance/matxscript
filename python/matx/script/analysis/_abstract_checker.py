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
import inspect

from matx._typed_ast import ast
from ..reporter import Reporter
from ..context import Span
from ...contrib.inspect3_9_1_patch import getsourcelines, findsource, getabsfile
from .source_analysis import dedent_and_margin, FixColOffset


class AbstractChecker(ast.NodeVisitor):

    def __init__(self):
        pass

    def is_pure_interface(self, cls_ty):
        if not inspect.isclass(cls_ty):
            return False
        for base in cls_ty.__bases__:
            if not isinstance(base, (abc.ABC, abc.ABCMeta)):
                return False
            # TODO: fix metaclass=ABCMeta
        lines, start_lineno = getsourcelines(cls_ty)
        src = ''.join(lines)
        new_src, offset = dedent_and_margin(src)
        root = ast.parse(new_src)
        root = FixColOffset(offset).visit(root)
        if isinstance(root, ast.Module) and len(root.body) == 1:
            root = root.body[0]
        if not isinstance(root, ast.ClassDef):
            return False
        source_code = ''.join(findsource(cls_ty)[0])
        self.span = Span()
        self.span.lineno = start_lineno
        self.span.source_code = source_code
        self.span.file_name = getabsfile(cls_ty)
        self.span.func_name = cls_ty.__name__
        self.raw_cls_ty = cls_ty
        return self.visit(root)

    def visit_ClassDef(self, node: ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str):
                pass
            elif isinstance(stmt, ast.FunctionDef):
                if len(stmt.decorator_list) != 1:
                    return False
                if (stmt.decorator_list[0].id != 'abstractmethod'
                        and stmt.decorator_list[0].id != 'abc.abstractmethod'):
                    return False
                for body_stmt in stmt.body:
                    if (not isinstance(body_stmt, ast.Pass)
                            and not (isinstance(stmt, ast.Expr)
                                     and isinstance(stmt.value, ast.Str))):
                        return False
            else:
                return False
        return True
