#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
import ast
import matx.kernel.graphIR as _gir


class BodyIterator:

    def __init__(self, node_stack, auto_add_return=False):
        self.body = []
        self.last_ast = None
        self.node_stack = node_stack
        self.auto_add_return = auto_add_return
        self.visited_added_return = False

    def has_next(self) -> bool:
        if not self.auto_add_return:
            return len(self.node_stack[-1]) > 0
        if self.visited_added_return:
            return False
        if len(self.node_stack[-1]) > 0:
            return True
        return self.auto_add_return and (
            len(self.body) == 0 or not isinstance(self.last_ast, ast.Return))

    def next(self):
        if len(self.node_stack[-1]) > 0:
            self.last_ast = self.node_stack[-1].pop()
            return self.last_ast
        self.visited_added_return = True
        return ast.Return(value=None)

    def push_ir(self, res):
        if res is not None:
            if not isinstance(res, _gir.Node):
                raise SyntaxError('Every IR node here should be a graphIR node!')
            self.body.append(res)
        else:
            # ignore the stmt
            pass
