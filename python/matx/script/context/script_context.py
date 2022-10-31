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

from ast import AST
from typing import Dict, List, Tuple, Optional, Any
from .ast_node import ASTNode
from .build_type import BuildType

from ... import ir as _ir
from ... import runtime as _rt
from ..reporter import Reporter


class ScriptContext:

    def __init__(self):
        self.build_type: Optional[BuildType] = None
        self.reporter = Reporter()
        self.main_node: ASTNode = ASTNode()
        self.deps_node: List[ASTNode] = []
        self.deps_relation: Dict[ASTNode, Tuple[ASTNode, AST]] = {}
        self.supported_imports = None
        self.builtin_symbols = None
        self.extern_symbols = None
        self.ir_module: Optional[_ir.IRModule] = None
        self.rt_module: Optional[_rt.Module] = None
        self.dso_path: Optional[Tuple[str, str]] = None
        self.free_vars: Optional[List[Any]] = None

    def get_ast_node_by_class_type(self, class_type: _ir.ClassType):
        if class_type.same_as(self.main_node.ir_schema):
            return self.main_node
        for dep in self.deps_node:
            if class_type.same_as(dep.ir_schema):
                return dep
        return None

    def add_free_var(self, var_ins):
        if self.free_vars is None:
            self.free_vars = [var_ins]
        else:
            self.free_vars.append(var_ins)
