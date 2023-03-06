#  // Copyright 2023 ByteDance Ltd. and/or its affiliates.
#  /*
#   * Licensed to the Apache Software Foundation (ASF) under one
#   * or more contributor license agreements.  See the NOTICE file
#   * distributed with this work for additional information
#   * regarding copyright ownership.  The ASF licenses this file
#   * to you under the Apache License, Version 2.0 (the
#   * "License"); you may not use this file except in compliance
#   * with the License.  You may obtain a copy of the License at
#   *
#   *   http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing,
#   * software distributed under the License is distributed on an
#   * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   * KIND, either express or implied.  See the License for the
#   * specific language governing permissions and limitations
#   * under the License.
#   */


from functools import partial

from repository import OpReplacementRepo
from .base_op import *
from .utils import *


class ArithmeticBinaryOp(KernelBaseOp):

    def __init__(self, lhs_type, rhs_type):
        super().__init__()
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type
        self.lhs_dtype = get_dtype(lhs_type)
        self.rhs_dtype = get_dtype(rhs_type)
        self.lhs_shape = get_shape(lhs_type)
        self.rhs_shape = get_shape(rhs_type)
        self.result_dtype = self._result_dtype()
        result_shape, lhs_new_shape, rhs_new_shape = broadcast(self.lhs_shape, self.rhs_shape)
        self.result_shape = result_shape
        self.lhs_broad_cast_shape = lhs_new_shape
        self.rhs_broad_cast_shape = rhs_new_shape

    def _result_dtype(self):
        result_dtype = np_result_dtype([self.lhs_dtype, self.rhs_dtype])
        return result_dtype


class AddOp(ArithmeticBinaryOp):
    opname = 'Add'
    operator = '+'


class SubOp(ArithmeticBinaryOp):
    opname = 'Sub'
    operator = '-'


class MultOp(ArithmeticBinaryOp):
    opname = 'Mult'
    operator = '*'


class DivOp(ArithmeticBinaryOp):
    opname = 'Div'
    operator = '/'


def array_array_binary_op(lhs: str, rhs: str, op_class: ArithmeticBinaryOp.__class__):
    pass


def make_bin_op(op_class):
    def op(func):
        return partial(func, op_class=op_class)

    OpReplacementRepo.add_bin_operator(
        op(array_array_binary_op),
        'NDArray',
        'NDArray',
        op_class.opname)


make_bin_op(AddOp)
make_bin_op(SubOp)
make_bin_op(MultOp)
make_bin_op(DivOp)
