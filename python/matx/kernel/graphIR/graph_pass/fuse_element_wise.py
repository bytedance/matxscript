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
from typing import List
from collections import OrderedDict
from ..ops import ElementWiseOperator, CopyOperator
from ..graph import Tensor, Node, IntVar, IntImm, Operator, Scalar
from queue import Queue


class FusedElementWiseOperator(ElementWiseOperator):

    def __init__(self, op_types, input_list: List[Tensor], output_list: List[Tensor],
                 sub_graph_input, sub_graph_nodes, sub_graph_outputs,
                 op1: ElementWiseOperator, op2: ElementWiseOperator):
        super().__init__()
        self.op_types = op_types
        self.result_dtype = op2.result_dtype
        self.result_shape = op2.result_shape
        self.results = output_list
        self._attrs["inputs"] = input_list
        self.op1 = op1
        self.op2 = op2
        self.sub_graph_input = sub_graph_input
        self.sub_graph_nodes = sub_graph_nodes
        self.sub_graph_outputs = sub_graph_outputs

    def draw_sub_graph(self):
        from ..utils import draw_graph
        draw_graph(self.sub_graph_nodes)


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
        if not isinstance(node, Tensor) or node.is_a_const_num() or node in self.graph_input:
            return
        src_ops = self._check_src_op(node)
        current_op = src_ops[0]
        if not isinstance(current_op, ElementWiseOperator):
            input_list = current_op._attrs["inputs"]
            for i in input_list:
                self._visit_tensor(i)
            return
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
            if isinstance(input_src_op, ElementWiseOperator):
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

    def _fuse_two_element_wise_op(
            self,
            op1,
            op2,
            tmp_tensor1: Tensor,
            tmp_tensor2: Tensor):
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
        if op1.is_scalar_op() or op2.is_scalar_op():
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
        # todo check if there are more tensor inbetween the two ops
        op_types: List = op1.op_types + op2.op_types
        inputs = op1._attrs["inputs"] + op2._attrs["inputs"]
        inputs.remove(tmp_tensor1)
        inputs = list(set(inputs))
        outputs = op1.get_outputs() + op2.get_outputs()
        outputs.remove(tmp_tensor1)
        fused_sub_graph = self._fuse_subgraph(op1, op2, tmp_tensor1)
        fused_op = FusedElementWiseOperator(op_types, inputs, outputs, *fused_sub_graph, op1, op2)

        # set src of tmp_tensor2 to the fused_op
        tmp_tensor2._attrs["src_ops"] = {fused_op}
        # set dst of op1's inputs to fused_op
        for op1_input in op1._attrs["inputs"]:
            op1_input._attrs["dst_ops"].remove(op1)
            op1_input._attrs["dst_ops"].add(fused_op)
        # set dst of op2's inputs to fused_op
        for op2_input in op2._attrs["inputs"]:
            op2_input._attrs["dst_ops"].remove(op2)
            op2_input._attrs["dst_ops"].add(fused_op)

        # remove op1, tmp_tensor1, and op2 from the graph
        self.graph_nodes.remove(op1)
        self.graph_nodes.remove(op2)
        self.graph_nodes.remove(tmp_tensor1)
        self.graph_nodes.append(fused_op)
        return fused_op

    def _fuse_subgraph(
            self,
            op1: ElementWiseOperator,
            op2: ElementWiseOperator,
            intermediate: Tensor):
        fused_sub_graph_input: OrderedDict[Tensor, Scalar] = OrderedDict()
        fused_sub_graph_nodes: List[Node] = []
        fused_sub_graph_outputs: OrderedDict[Scalar, Tensor] = OrderedDict()
        # update inputs
        fused_sub_graph_input.update(op1.sub_graph_input)
        for k, v in op2.sub_graph_input.items():
            if k == intermediate:
                continue
            if k in fused_sub_graph_input:
                op1_v = fused_sub_graph_input[k]
                copy_op = CopyOperator()
                copy_op(v, op1_v)
                fused_sub_graph_nodes.append(copy_op)
                continue
            fused_sub_graph_input[k] = v
        # update nodes
        output_scalar = None
        for node in op1.sub_graph_nodes:
            if (node in op1.sub_graph_outputs) and (op1.sub_graph_outputs[node] == intermediate):
                output_scalar = node
            fused_sub_graph_nodes.append(node)
        for node in op2.sub_graph_nodes:
            if op2.sub_graph_input[intermediate] == node:
                if output_scalar is None:
                    raise SyntaxError("got error during fusing element ops")
                ops = node._attrs["dst_ops"]
                output_scalar._attrs["dst_ops"].update(ops)
                for op in ops:
                    op_inputs = op._attrs["inputs"]
                    idx = op_inputs.index(node)
                    op._attrs["inputs"][idx] = output_scalar
                output_scalar = None
                continue
            fused_sub_graph_nodes.append(node)
        # update outputs
        fused_sub_graph_outputs.update(
            {k: v for k, v in op1.sub_graph_outputs.items() if v != intermediate})
        fused_sub_graph_outputs.update(op2.sub_graph_outputs)
        return fused_sub_graph_input, fused_sub_graph_nodes, fused_sub_graph_outputs
