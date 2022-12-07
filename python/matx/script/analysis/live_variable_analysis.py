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
from ..._typed_ast import ast
from ... import cfg


class LiveVariableAnalysis:

    def __init__(self, ast_func_def: ast.FunctionDef):
        my_cfg = cfg.CFG(ast_tree=ast_func_def)
        my_cfg.fill_dominance_frontier()
        my_cfg.init_var_life_info()
        my_cfg.compute_live_out_var()
        my_cfg.compute_reaching_defines()
        self._live_out_mapping = my_cfg.collect_ast_live_out_info()

    def get_live_out_mapping(self):
        return self._live_out_mapping
