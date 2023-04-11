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


from numbers import Number

from matx.ir.tensor_stmt import BufferRegion
from matx.kernel.typing import *


def get_dtype(t):
    if isinstance(t, NDArrayType):
        return t.dtype
    elif isinstance(t, (Number, np.bool_)):
        return PYTYPE_TO_KERNEL_TYPE[type(t)]
    else:
        raise TypeError(f"Type {type(t)} of argument {t} is not supported")


def get_shape(t):
    if isinstance(t, NDArrayType):
        return t.shape
    elif isinstance(t, (Number, np.bool_)):
        return None
    else:
        raise TypeError(f"Shape {type(t)} of argument {t} is not supported")


def np_result_dtype(nptypes):
    restype = np.result_type(*nptypes)
    if restype.type not in PYTYPE_TO_KERNEL_TYPE.keys():
        for k in PYTYPE_TO_KERNEL_TYPE.keys():
            if k == restype.type:
                return k
    return restype.type


def make_buffer_region(buffer, range):
    return BufferRegion(buffer, range)
