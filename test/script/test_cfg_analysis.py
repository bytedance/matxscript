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
import inspect
import unittest
import re
import matx

from matx import cfg
from matx.contrib.inspect3_9_1_patch import getsourcelines
from typed_ast import ast3 as ast


def dedent_and_margin(text):
    whitespace_only = re.compile('^[ \t]+$', re.MULTILINE)
    leading_whitespace = re.compile('(^[ \t]*)(?:[^ \t\n])', re.MULTILINE)
    margin = None
    text = whitespace_only.sub('', text)
    indents = leading_whitespace.findall(text)
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


def get_ast_tree(fn):
    source_code = inspect.getsource(fn)
    new_src, offset = dedent_and_margin(source_code)
    as_tree = ast.parse(new_src)
    assert isinstance(as_tree, ast.Module)
    return as_tree.body[0]


def get_ast_and_cfg(fn):
    ast_tree = get_ast_tree(fn)
    lines, start_lineno = getsourcelines(fn)
    my_cfg = cfg.CFG(ast_tree=ast_tree)
    my_cfg.fill_dominance_frontier()
    my_cfg.init_var_life_info()
    my_cfg.compute_live_out_var()
    return start_lineno, ast_tree, my_cfg


class TestControlFlowGraphAnalysis(unittest.TestCase):

    def test_basic_if(self):
        def my_fn():
            a = 3
            if a > 3:
                a = 98
                z = 5
            else:
                z = 4
                a = 100
            y = z
            h = a

        lineno, tree, my_cfg = get_ast_and_cfg(my_fn)
        print(my_cfg)
        self.assertEqual(len(my_cfg.block_list), 5)
        start_end_lines = [(blk.start_line, blk.end_line) for blk in my_cfg.block_list]
        self.assertEqual(start_end_lines, [(1, 1), (2, 3), (4, 5), (7, 8), (9, 10)])
        live_outs = [blk.live_out for blk in my_cfg.block_list]
        self.assertEqual(live_outs, [set(), set(), {'a', 'z'}, {'a', 'z'}, set()])


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
