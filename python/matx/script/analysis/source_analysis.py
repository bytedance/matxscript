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

import re
from matx._typed_ast import ast

from .. import context
from ..reporter import raise_syntax_error
from ...contrib.inspect3_9_1_patch import getsourcelines, findsource, getabsfile
import inspect

_whitespace_only_re = re.compile('^[ \t]+$', re.MULTILINE)
_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])', re.MULTILINE)


def dedent_and_margin(text):
    # mainly from textwrap.dedent
    margin = None
    text = _whitespace_only_re.sub('', text)
    indents = _leading_whitespace_re.findall(text)
    for indent in indents:
        if margin is None:
            margin = indent

        elif indent.startswith(margin):
            pass

        elif margin.startswith(indent):
            margin = indent

        else:
            for i, (x, y) in enumerate(zip(margin, indent)):
                if x != y:
                    margin = margin[:i]
                    break

    if margin is not None:
        text = re.sub(r'(?m)^' + margin, '', text)
    else:
        margin = ''
    return text, len(margin)


class FixColOffset(ast.NodeTransformer):
    def __init__(self, offset) -> None:
        self.offset = offset
        super().__init__()

    def visit(self, node):
        """Visit a node."""
        if hasattr(node, 'col_offset'):
            node.col_offset += self.offset
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)


class SourceAnalysis:

    def __init__(self) -> None:
        self.change = False
        self.sc_ctx = None

    def run_impl(self, node: context.ASTNode):
        if node.ast is None:
            try:
                lines, start_lineno = getsourcelines(node.raw)
            except Exception as e:
                dep_relation = self.sc_ctx.deps_relation.get(node)
                if dep_relation is None:
                    raise OSError("Can't get source codes of {}".format(node.raw))
                from_node, ast_node = dep_relation
                raise_syntax_error(
                    from_node,
                    ast_node,
                    "Can't get source codes of {}".format(
                        node.raw),
                    type(e))
            src = ''.join(lines)
            new_src, offset = dedent_and_margin(src)
            root = ast.parse(new_src)
            root = FixColOffset(offset).visit(root)
            if isinstance(root, ast.Module) and len(root.body) == 1:
                root = root.body[0]
            node.ast = root
            node.span.source_code = ''.join(findsource(node.raw)[0])
            node.span.lineno = start_lineno
            node.span.file_name = getabsfile(node.raw)
            node.span.func_name = node.raw.__name__
            node.extra["co_lines"] = len(lines)
            node.extra["co_chars"] = len(src)
            node.extra["is_class"] = inspect.isclass(node.raw)
            self.change = True

    def run(self, sc_ctx: context.ScriptContext):
        self.change = False
        self.sc_ctx = sc_ctx
        self.run_impl(sc_ctx.main_node)
        for dep_node in sc_ctx.deps_node:
            self.run_impl(dep_node)
        self.sc_ctx = None
        return self.change
