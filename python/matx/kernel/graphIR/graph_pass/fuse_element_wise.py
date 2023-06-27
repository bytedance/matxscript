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
from ..ops import ElementWiseOperator
from ..graph import Tensor, Scalar, Node, IntVar, IntImm, Operator
from queue import Queue


class FusedElementWiseOperator(ElementWiseOperator):

    def __init__(self, op_types, tensor_list: List[Tensor]):
        super().__init__()
        self.op_types = op_types
        self._attrs["inputs"] = tensor_list


class ElementWiseOpFuser:

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
        for node in graph_output:
            if isinstance(node, Tensor):
                self._visit_tensor(node)
            elif isinstance(node, IntVar):
                print("[warning] symbol as return")
                continue
            elif isinstance(node, IntImm):
                continue
            else:
                raise NotImplementedError(
                    f"value {node} has type {type(node)} as return, which is not supported")

    def _visit_tensor(self, node: Tensor):
        if node in self.visited:
            return
        self.visited.add(node)
        if node.is_a_const_num() or node in self.graph_input:
            return
        src_ops = self._check_src_op(node)
        current_op = src_ops[0]
        node_queue = Queue()
        while (rt := self._visit_tensor_helper(node, current_op, node_queue)) is not None:
            src_ops = self._check_src_op(node)
            current_op = src_ops[0]
        while not node_queue.empty():
            self._visit_tensor(node_queue.get())

    def _visit_tensor_helper(self, current_node, current_op, node_queue: Queue):
        node_queue.empty()
        for input_tensor in current_op._attrs["inputs"]:
            if not isinstance(input_tensor, Tensor):
                continue
            if input_tensor.is_a_const_num() or input_tensor in self.graph_input:
                continue
            input_src_op = self._check_src_op(input_tensor)[0]
            result = self._fuse_two_element_wise_op(
                input_src_op, current_op, input_tensor, current_node)
            if result is not None:
                return result
            node_queue.put(input_tensor)
        return None

    def _check_src_op(self, node: Tensor) -> List[Operator]:
        src_ops = list(node.src_ops())
        if (not node.is_a_const_num()) and len(src_ops) != 1:
            raise SyntaxError(f"The tensor {node} has {len(src_ops)} src ops, which is likely "
                              f"due to Erroneously constructed graph")
        return src_ops

    def _fuse_two_element_wise_op(self, op1: ElementWiseOperator, op2: ElementWiseOperator,
                                  tmp_tensor1: Tensor, tmp_tensor2: Tensor):
        """
        Before:
        a:tensor -\
                   -op1(+) -> tmp_tensor1:tensor ------\
        b:tensor -/                                     - op2(-) -> tmp_tensor2:tensor
        c:tensor --------------------------------------/

        After:
        a:tensor --\
        b:tensor -- -fused_op(+, -) -> tmp_tensor2:tensor
        c:tensor --/

        Parameters
        ----------
        op1
        op2
        tmp_tensor1
        tmp_tensor2

        Returns
        -------

        """
        if not (isinstance(op1, ElementWiseOperator) and isinstance(op2, ElementWiseOperator)):
            return None

        tmp1_src = tmp_tensor1.src_ops()
        tmp1_dst = tmp_tensor1.dst_ops()
        tmp2_src = tmp_tensor2.src_ops()
        if len(tmp1_src) != 1 or tuple(tmp1_src)[0] != op1:
            return None
        if len(tmp1_dst) != 1 or tuple(tmp1_dst)[0] != op2:
            return None
        if len(tmp2_src) != 1 or tuple(tmp2_src)[0] != op2:
            return None
        op_types: List = op1.op_types + op2.op_types
        inputs = op1._attrs["inputs"] + op2._attrs["inputs"]
        inputs.remove(tmp_tensor1)
        fused_op = FusedElementWiseOperator(op_types, inputs)
        # set src of tmp_tensor2 to the fused_op
        tmp_tensor2._attrs["src_ops"] = {fused_op}
        # set dst of op1's inputs to fused_op
        for op1_input in op1._attrs["inputs"]:
            op1_input._attrs["dst_ops"].remove(op1)
            op1_input._attrs["dst_ops"].add(fused_op)
        # remove op1, tmp_tensor1, and op2 from the graph
        self.graph_nodes.remove(op1)
        self.graph_nodes.remove(op2)
        self.graph_nodes.remove(tmp_tensor1)
        self.graph_nodes.append(fused_op)
        return fused_op
