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

import matx.kernel.graphIR.utils as graph_utils
import matx.kernel.typing.utils as typing_utils
from matx.kernel.graphIR import Operator, Tensor


class ElementWiseOperator(Operator):

    def __init__(self):
        super().__init__()
        self.op_types = []

    def __call__(self, *args: List[Tensor]) -> List[Tensor]:
        pass


class FusedElementWiseOperator(ElementWiseOperator):

    def __init__(self, op_types, tensor_list: List[Tensor]):
        super().__init__()
        self.op_types = op_types
        self._attrs["inputs"] = tensor_list

    def __call__(self, *args, **kwargs):
        pass


class BinaryElementWiseOperator(ElementWiseOperator):

    def __init__(self, op_type):
        super().__init__()
        self.op_types = [op_type]

    def __call__(self, lhs: Tensor, rhs: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [lhs, rhs]
        lhs.dst_ops().add(self)
        rhs.dst_ops().add(self)
        lhs_dtype = lhs.dtype()
        lhs_shape = lhs.shape()
        rhs_dtype = rhs.dtype()
        rhs_shape = rhs.shape()
        # todo dtype conversion
        result_dtype = typing_utils.np_result_dtype([lhs_dtype, rhs_dtype])
        result_shape = graph_utils.broadcast(lhs_shape, rhs_shape)
        result_tensor = Tensor(result_shape, src_ops=[self], dtype=result_dtype)
        return [result_tensor]


class UnaryElementWiseOperator(ElementWiseOperator):

    def __init__(self, op_type):
        super().__init__()
        self.op_types = [op_type]

    def __call__(self, operand: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [operand]
        pass


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

    def __call__(self, copy_to: Tensor, copy_from: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [copy_to, copy_from]
        # read from
        copy_from.dst_ops().add(self)
        # write to
        copy_to.src_ops().add(self)
        return [copy_to]


class DeepCopyOperator(CopyOperator):

    def __init__(self):
        super().__init__()

    def __call__(self, copy_to: Tensor, copy_from: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [copy_to, copy_from]
        return super().__call__(copy_to, copy_from)
