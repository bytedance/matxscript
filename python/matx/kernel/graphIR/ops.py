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
import numpy as np
from typing import List, Set
from collections import OrderedDict
import matx.kernel.graphIR.utils as graph_utils
import matx.kernel.typing.utils as typing_utils
from matx.kernel.graphIR import Operator, Tensor, Node, Scalar


def not_tensor(x):
    return not (isinstance(x, Tensor) and len(x.shape()) != 0)


class ElementWiseOperator(Operator):

    def __init__(self):
        super().__init__()
        self.op_types = []
        self.result_dtype = None
        self.result_shape = None
        self.results = []
        self.sub_graph_input: OrderedDict[Tensor, Scalar] = OrderedDict()
        self.sub_graph_nodes: List[Node] = []
        self.sub_graph_outputs: OrderedDict[Scalar, Tensor] = OrderedDict()

    def __call__(self, *args: List[Tensor]) -> List[Tensor]:
        pass

    def get_inputs(self):
        return self._attrs["inputs"]

    def get_outputs(self):
        return self.results

    def is_scalar_op(self):
        return all(not_tensor(i) for i in self.get_inputs())

    def make_subgraph(self, f):
        if self.is_scalar_op():
            return
        inputs = self.get_inputs()
        for i in inputs:
            if not_tensor(i):
                if i not in self.sub_graph_input:
                    self.sub_graph_input[i] = i
            elif i not in self.sub_graph_input:
                scalar = Scalar(name=i.name() + "_scalar", dtype=i.dtype())
                self.sub_graph_input[i] = scalar

        for i in self.results:
            if i not in self.sub_graph_outputs:
                scalar = Scalar(name=i.name() + "_scalar", dtype=i.dtype())
                self.sub_graph_outputs[scalar] = i
        self.sub_graph_nodes = [
            *self.sub_graph_input.values(),
            *f(self.sub_graph_input, self.sub_graph_outputs),
            *self.sub_graph_outputs
        ]


class BinaryElementWiseOperator(ElementWiseOperator):

    def __init__(self, op_type):
        super().__init__()
        self.op_types = [op_type]
        self.is_boolean_op = op_type in (
            ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.And, ast.Or)

    def _sub_graph_maker(self, ins: OrderedDict[Tensor, Scalar], outs: OrderedDict[Scalar, Tensor]):
        bin_op = BinaryElementWiseOperator(self.op_types[0])
        in_values = list(ins.values())
        bin_result = bin_op(in_values[0], in_values[1])[0]
        copy_op = CopyOperator()
        out_values = list(outs.keys())
        out = out_values[0]
        copy_op(out, bin_result)
        return [bin_op, bin_result, copy_op]

    def __call__(self, lhs: Tensor, rhs: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [lhs, rhs]
        lhs.dst_ops().add(self)
        rhs.dst_ops().add(self)
        self.lhs_dtype = lhs.dtype()
        self.lhs_shape = lhs.shape()
        self.rhs_dtype = rhs.dtype()
        self.rhs_shape = rhs.shape()

        if self.is_boolean_op:
            self.result_dtype = typing_utils.convert_to_string_dtype(np.bool_)
        elif self.op_types[0] == ast.Div:
            lhs_np_dtype = typing_utils.STR_TO_PYTYPE[self.lhs_dtype]
            rhs_np_dtype = typing_utils.STR_TO_PYTYPE[self.rhs_dtype]
            result_np_dtype = (lhs_np_dtype(1) / rhs_np_dtype(1)).dtype.type
            self.result_dtype = typing_utils.PYTYPE_TO_STR[result_np_dtype]
        else:
            self.result_dtype = typing_utils.convert_to_string_dtype(
                typing_utils.np_result_dtype([self.lhs_dtype, self.rhs_dtype]))
        self.result_shape = graph_utils.broadcast(self.lhs_shape, self.rhs_shape)
        result_tensor = Tensor(self.result_shape, src_ops=[self], dtype=self.result_dtype)
        self.results = [result_tensor]
        self.make_subgraph(self._sub_graph_maker)
        return self.results


class UnaryElementWiseOperator(ElementWiseOperator):

    def __init__(self, op_type):
        super().__init__()
        self.op_types = [op_type]
        self.is_boolean_op = op_type in (
            ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.And, ast.Or)

    def _sub_graph_maker(self, ins: OrderedDict[Tensor, Scalar], outs: OrderedDict[Scalar, Tensor]):
        bin_op = UnaryElementWiseOperator(self.op_types[0])
        in_values = list(ins.values())
        bin_result = bin_op(in_values[0])[0]
        copy_op = CopyOperator()
        out_values = list(outs.keys())
        out = out_values[0]
        copy_op(out, bin_result)
        return [bin_op, bin_result, copy_op]

    def __call__(self, operand: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [operand]
        operand.dst_ops().add(self)
        self.operand_dtype = operand.dtype()
        self.operand_shape = operand.shape()

        if self.is_boolean_op:
            self.result_dtype = typing_utils.convert_to_string_dtype(np.bool_)
        else:
            self.result_dtype = typing_utils.convert_to_string_dtype(
                typing_utils.np_result_dtype([self.operand_dtype]))
        self.result_shape = self.operand_shape
        result_tensor = Tensor(self.result_shape, src_ops=[self], dtype=self.result_dtype)
        self.results = [result_tensor]
        self.make_subgraph(self._sub_graph_maker)
        return self.results


class ReductionOperator(Operator):
    pass


class CopyOperator(Operator):
    """
    This is a shallow copy operator.
    For tensors, shallow copy does not copy element.
    For scalars, it copies.
    """

    def __init__(self):
        super().__init__()
        self.copy_to = None
        self.copy_from = None

    def __call__(self, copy_to: Tensor, copy_from: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [copy_to, copy_from]
        self.copy_to = copy_to
        self.copy_from = copy_from
        # read from
        copy_from.dst_ops().add(self)
        # write to
        copy_to.src_ops().add(self)
        return [copy_to]


class DeepCopyOperator(CopyOperator):

    def __init__(self):
        super().__init__()

    def __call__(self, copy_to: Tensor, copy_from: Tensor) -> List[Tensor]:
        return super().__call__(copy_to, copy_from)





class IntVarCopyOperator(Node):

    def pseudo_code(self, with_shape: bool = False) -> str:
        return "IntVarCopyOperator"


    def __init__(self):
        super().__init__()
        self._attrs["inputs"] = None

    def __call__(self, *args, **kwargs):
        pass
