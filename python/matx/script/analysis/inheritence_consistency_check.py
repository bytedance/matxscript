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
from typing import Dict, List
from matx._typed_ast import ast
from inspect import getmro, isclass
from .. import context
from ..reporter import raise_syntax_error


def _get_raw_node_map(node: context.ASTNode):
    m: Dict[type, context.ASTNode] = {}
    q: List[context.ASTNode] = [node]
    visited = set()
    while q:
        n = q.pop(0)
        m[n.raw] = n
        for dep in n.deps:
            if dep.raw not in visited:
                q.append(dep)
                visited.add(dep.raw)
    return m


def _get_func_type_map(cls_ctx: context.ClassContext):
    func_type_map = {}
    for name, func in cls_ctx.methods.items():
        # function signatures don't involve the parameter names, only the types
        func_type_map[name] = {
            'arg_types': [func.arg_types[x] for x in func.arg_names],
            'ret_type': func.return_type,
            'defaults': func.arg_defaults
        }
    return func_type_map


class InheritencyConsistencyCheck(ast.NodeVisitor):

    def run_impl(self, node: context.ASTNode):
        if not isclass(node.raw):
            return
        raw_node_map = _get_raw_node_map(node)
        type_map = _get_func_type_map(node.context)
        type_map.update(node.context.attr_types)
        for base in getmro(node.raw):
            if base is node.raw:
                continue
            base_node = raw_node_map.get(base)
            if base_node is None:
                # It's not easy to list all ignore bases: object, ABC, pure_interface, torch.nn.Module...
                # This checker will not blame that it can't find dependency anymore but just ignore them.
                # assert base_node is not None, "base of {}: {} is not found in deps, it should be a bug of matxscript, please report it.".format(
                #     node.raw.__name__, base.__name__)
                continue
            base_type_map = _get_func_type_map(base_node.context)
            base_type_map.update(base_node.context.attr_types)
            for var, var_type in type_map.items():
                if var in base_type_map and var_type != base_type_map[var]:
                    raise_syntax_error(
                        node,
                        node.ast,
                        "{}.{} is decalared with type {}, which is not consistent to that in the base, {}.{}: {}".format(
                            node.raw.__name__,
                            var,
                            var_type,
                            base_node.raw.__name__,
                            var,
                            base_type_map[var]))

    def run(self, sc_ctx: context.ScriptContext):
        for dep_node in [sc_ctx.main_node] + sc_ctx.deps_node:
            self.run_impl(dep_node)
