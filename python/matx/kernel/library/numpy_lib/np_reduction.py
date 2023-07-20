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

from matx.kernel.kernel_parser import KernelParser
from jinja2 import Template
import matx.kernel.graphIR as _gir
from typing import List, Set
import matx.kernel.typing.utils as typing_utils
import ast
import numpy as np
from .LIB import _TEMPLATE_FUNCTION_DICT, _LIBRARY_NODE_DICT

np_max_template = Template("""
def np_max__{{in_type}}_{{in_shape|join("_")}}__{{out_type}}_{{out_shape|join("_")}}(a:{{in_type}}[{{in_shape|join(', ')}}], axis=None)->{{out_type}}[{{out_shape|join(', ')}}]:
""")


class Sum(_gir.ReductionOperator):

    def __call__(self, operand: _gir.Tensor) -> List[_gir.Tensor]:
        # use function
        self._attrs["inputs"] = [operand]
        operand_type = operand.dtype()
        operand_shape = operand.shape()

        np_dtype = typing_utils.STR_TO_PYTYPE[operand_type]
        result_np_dtype = np.sum(np.array([1, 2], dtype=np_dtype)).dtype.type
        result_dtype = typing_utils.PYTYPE_TO_STR[result_np_dtype]

        self.reduction_dims.extend(range(len(operand_shape)))
        sum_value = _gir.Scalar(dtype=result_dtype, value=0, dst_ops=[self])
        self.sub_graph_init_values.append(sum_value)
        sum_value_scalar = _gir.Scalar(dtype=result_dtype)
        self.sub_graph_input[sum_value] = sum_value_scalar
        self.sub_graph_nodes.append(sum_value_scalar)
        input_scalar = _gir.Scalar(dtype=operand_type)
        self.sub_graph_input[operand] = input_scalar
        self.sub_graph_nodes.append(input_scalar)
        add_op = _gir.BinaryElementWiseOperator(ast.Add)
        self.sub_graph_nodes.append(add_op)
        result_scalar = add_op(sum_value_scalar, input_scalar)[0]
        self.sub_graph_nodes.append(result_scalar)
        result = _gir.Scalar(dtype=result_dtype, src_ops=[self])
        self.sub_graph_outputs[result_scalar] = result
        self.results.append(result)
        return [result]


class Prod(_gir.ReductionOperator):

    def calcaulte_result_type(self, operand_type_str):
        np_dtype = typing_utils.STR_TO_PYTYPE[operand_type_str]
        result_np_dtype = np.prod(np.array([1, 2], dtype=np_dtype)).dtype.type
        result_dtype_str = typing_utils.PYTYPE_TO_STR[result_np_dtype]
        return result_dtype_str

    def __call__(self, operand: _gir.Tensor) -> List[_gir.Tensor]:
        # use function
        self._attrs["inputs"] = [operand]
        operand_type_str = operand.dtype()
        operand_shape = operand.shape()
        operand_kernel_t = _gir.utils.convert_to_kernel_type(operand)
        operand_dtype_kernel_t = operand_kernel_t.data_type()
        result_dtype_str = self.calcaulte_result_type(operand_type_str)
        result_kernel_t = typing_utils.STR_TO_KERNEL_TYPE[result_dtype_str]

        def sub_graph_func(
                init_value: result_kernel_t,
                a: operand_dtype_kernel_t) -> result_kernel_t:
            return init_value * a

        p = KernelParser(sub_graph_func)
        p.parse()
        sub_graph_input = p.graph.graph_input
        sub_graph_output = p.graph.graph_output
        sub_graph_nodes = p.graph.graph_nodes

        self.reduction_dims.extend(range(len(operand_shape)))
        sum_value = _gir.Scalar(dtype=result_dtype_str, value=1, dst_ops=[self])
        self.sub_graph_init_values.append(sum_value)

        self.sub_graph_input[sum_value] = sub_graph_input[0]
        self.sub_graph_input[operand] = sub_graph_input[1]
        result = _gir.Scalar(dtype=result_dtype_str, src_ops=[self])
        self.sub_graph_outputs[sub_graph_output[0]] = result
        self.sub_graph_nodes.extend(sub_graph_output)
        self.results.append(result)
        return [result]


_LIBRARY_NODE_DICT["sum"] = Sum
_LIBRARY_NODE_DICT["prod"] = Prod
