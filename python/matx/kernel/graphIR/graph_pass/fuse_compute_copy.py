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


class TmpVarEliminator:
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

        for node in graph_nodes:
            if isinstance(node, CopyOperator):
                self.remove(node)

    def remove(self, node: CopyOperator):
        """
        Before:

        ---> copy_from:tensor -\
                                - copy_op() -> copy_to:tensor -->
             copy_to:tensor ---/

        After:

        ---> copy_to:tensor -->



        Parameters
        ----------
        node

        Returns
        -------

        """

        from_node = node.copy_from
        to_node = node.copy_to

        if (not isinstance(from_node, Tensor)) and (not isinstance(to_node, Tensor)):
            return None

        if to_node not in self.graph_input and to_node not in self.graph_output:
            return self._remove_to_node(node, from_node, to_node)
        elif from_node not in self.graph_input and from_node not in self.graph_output:
            return self._remove_from_node(node, from_node, to_node)
        else:
            return None

    def _remove_from_node(self, copy_op, from_node, to_node):
        copy_from_src_ops = from_node.src_ops()
        for op in copy_from_src_ops:
            if not isinstance(op, (ElementWiseOperator,)):
                return None
        self.graph_nodes.remove(copy_op)
        self.graph_nodes.remove(from_node)
        # update to_node src ops
        copy_to_src_ops = to_node.src_ops()
        copy_to_src_ops.update(copy_from_src_ops)
        to_node._attrs["src_ops"] = copy_to_src_ops
        to_node._attrs["src_ops"].remove(copy_op)
        # update src ops outputs
        for op in copy_from_src_ops:
            idx = op.results.index(from_node)
            if idx != -1:
                op.results[idx] = to_node

            for scalar, tensor in op.sub_graph_outputs.items():
                if tensor == from_node:
                    op.sub_graph_outputs[scalar] = to_node

    def _remove_to_node(self, copy_op, from_node, to_node):
        copy_to_dst_ops = to_node.dst_ops()
        self.graph_nodes.remove(copy_op)
        self.graph_nodes.remove(to_node)
        # update from_op dst ops
        copy_from_dst_ops = from_node.dst_ops()
        copy_from_dst_ops.update(copy_to_dst_ops)
        from_node._attrs["dst_ops"] = copy_from_dst_ops
        from_node._attrs["dst_ops"].remove(copy_op)
        # update dst ops inputs
        for op in copy_to_dst_ops:
            if to_node in op._attrs["inputs"]:
                idx = op._attrs["inputs"].index(to_node)
                if idx != -1:
                    if from_node not in op._attrs["inputs"]:
                        op._attrs["inputs"][idx] = from_node
                    else:
                        op._attrs["inputs"].pop(idx)
            if isinstance(op, ElementWiseOperator):
                new_inputs: OrderedDict[Tensor, Scalar] = OrderedDict()
                for k, v in op.sub_graph_input.items():
                    new_key = k
                    if k == to_node:
                        new_key = from_node
                    if new_key in new_inputs:
                        v2 = new_inputs[new_key]
                        copy_op = CopyOperator()
                        copy_op(v2, v)
                        op.sub_graph_nodes.append(copy_op)
                    new_inputs[new_key] = v
                op.sub_graph_input = new_inputs
