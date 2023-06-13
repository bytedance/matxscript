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

import numpy as np

from ..symbol import is_symbol

NPDTYPE_TO_STR = {
    np.bool_: 'bool',
    np.int8: 'int8',
    np.int16: 'int16',
    np.int32: 'int32',
    np.int64: 'int64',
    np.intc: 'int32',
    np.uint8: 'uint8',
    np.uint16: ' uint16',
    np.uint32: ' uint32',
    np.uint64: ' uint64',
    np.uintc: 'uint32',
    np.float16: 'float16',
    np.float32: 'float32',
    np.float64: 'float64',
    np.longlong: 'int64',
    np.ulonglong: ' uint64'
}


class NDArrayType:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.storage = 'cpu'
        self.symbol_list = [axis for axis in shape if is_symbol(axis)]

    def __repr__(self):
        return f'NDArrayType (dtype={self.dtype}, shape={self.shape})'

    def dtype_str(self):
        return NPDTYPE_TO_STR[self.dtype]

    def data_type(self):
        return ScalarType(self.dtype)

    def __eq__(self, other):
        return self.shape == other.shape and self.dtype == self.dtype

    def compatible_with(self, other):
        # todo check dtype
        return self.shape == other and self.dtype == self.dtype


class ScalarType(NDArrayType):

    def __init__(self, dtype):
        super().__init__((1, ), dtype)

    def __getitem__(self, shape) -> NDArrayType:
        if isinstance(shape, list) or isinstance(shape, tuple):
            if list(shape) == [1]:
                return self
            return NDArrayType(tuple(shape), self.dtype)
        if shape == 1:
            return self
        return NDArrayType((shape,), self.dtype)

    def __repr__(self) -> str:
        return f'ScalarType (dtype={self.dtype}, storage={self.storage})'

    def data_type(self):
        return self
