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


class BuildTypeAnalysis:

    def __init__(self) -> None:
        self.change = False

    def run(self, sc_ctx: context.ScriptContext):
        self.change = False
        node_ctx = sc_ctx.main_node.context
        if isinstance(node_ctx, context.ClassContext):
            build_type = context.BuildType.JIT_OBJECT
        elif isinstance(node_ctx, context.FunctionContext):
            build_type = context.BuildType.FUNCTION
        else:
            raise RuntimeError("Only one-function, one-class source code is allowed")
        if sc_ctx.build_type != build_type:
            self.change = True
            sc_ctx.build_type = build_type
        return self.change
