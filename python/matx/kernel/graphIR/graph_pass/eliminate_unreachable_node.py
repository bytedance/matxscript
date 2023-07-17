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
from typing import List, Union
from collections import OrderedDict
from ..ops import CopyOperator, ElementWiseOperator
from ..graph import Tensor, Scalar, Node, IntVar, IntImm, Operator
from queue import Queue


class UnreachableNodeEliminator:

    def __init__(self):
        self.graph_input: List[Node] = []
        self.graph_output: List[Node] = []
        self.graph_nodes: List[Node] = []
        self.visited = set()

    def apply(self, graph_input: List[Node], graph_output: List[Node], graph_nodes: List[Node]):
        self.graph_input = graph_input
        self.graph_output = graph_output
        self.graph_nodes = graph_nodes
        self.visited = set()
        for out in self.graph_output:
            self._visit(out)

        for i, v in enumerate(self.graph_output):
            if v not in self.visited:
                self.graph_output.pop(i)

    def _visit(self, node):
        if not isinstance(node, Tensor):
            return
        if node in self.visited:
            return
        self.visited.add(node)
        for op in node._attrs["src_ops"]:
            for op_in in op._attrs["inputs"]:
                self._visit(op_in)
