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

from .. import context

from .source_analysis import SourceAnalysis
from .module_analysis import ModuleAnalysis
from .deps_analysis import DepsAnalysis
from .function_analysis import FunctionAnalysis
from .class_analysis import ClassAnalysis
from .build_type_analysis import BuildTypeAnalysis
from .syntax_check import SyntaxCheck
from .inheritence_consistency_check import InheritencyConsistencyCheck

from .live_variable_analysis import LiveVariableAnalysis


class CallableAnalysis:
    """
    Unify the run of class/function analysis
    """

    @classmethod
    def get_node_analysis(cls, node: context.ASTNode):
        analysis = None
        if isinstance(node.ast, ast.ClassDef):
            analysis = ClassAnalysis()
        elif isinstance(node.ast, ast.FunctionDef):
            analysis = FunctionAnalysis()
        else:
            raise Exception(f'Invalid node: {node.ast}')
        return analysis

    def _run(self, sc_ctx: context.ScriptContext, action: str):
        for dep_node in reversed(sc_ctx.deps_node):
            self.get_node_analysis(dep_node).run_impl(dep_node, sc_ctx, action=action)
        self.get_node_analysis(sc_ctx.main_node).run_impl(sc_ctx.main_node, sc_ctx, action=action)

    def run(self, sc_ctx: context.ScriptContext):
        # Step 1: Init context
        self._run(sc_ctx, 'INIT')

        # Step 2: Parse type
        self._run(sc_ctx, 'TYPE')
