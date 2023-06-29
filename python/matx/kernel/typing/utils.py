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


import numbers

import numpy as np
import sympy

from matx.kernel.symbol.utils import is_symbol
from matx.kernel.typing.kernel_type import NDArrayType
from matx.kernel.typing.type_def import *


def is_scalar(x: NDArrayType):
    return is_scalar_shape(x.shape)


def is_scalar_shape(shape):
    if is_symbol(shape[0]):
        return False
    if isinstance(shape[0], numbers.Number):
        return shape[0] == 1
    if len(shape) == 1:
        return is_scalar_shape(shape[0])
    return False


def is_ndarray_type(t):
    return t is NDArrayType or isinstance(t, NDArrayType)


def is_scalar_type(t):
    if is_ndarray_type(t):
        return is_scalar_shape(t.shape)
    return t is ScalarType or isinstance(t, ScalarType)


def convert_to_np_dtype(x):
    if x in PYTYPE_TO_STR:
        return x
    if isinstance(x, bool):
        return np.bool_
    if isinstance(x, int):
        return np.int64
    if isinstance(x, float):
        return np.float32


def convert_to_string_dtype(x):
    np_dtype = convert_to_np_dtype(x)
    return PYTYPE_TO_STR[np_dtype]


def get_dtype_str(t):
    if isinstance(t, NDArrayType):
        return t.dtype_str()
    elif isinstance(t, (numbers.Number, np.bool_)):
        return PYTYPE_TO_KERNEL_TYPE[type(t)].dtype_str()
    elif t is sympy.Basic:
        return "int64"
    else:
        raise TypeError(f"Type {type(t)} of argument {t} is not supported")


def get_shape(t):
    if isinstance(t, NDArrayType):
        return t.shape
    elif isinstance(t, (numbers.Number, np.bool_)):
        return None
    elif t is sympy.Basic:
        return (1,)
    else:
        raise TypeError(f"Shape {type(t)} of argument {t} is not supported")


def np_result_dtype(nptypes):
    restype = np.result_type(*nptypes)
    if restype.type not in PYTYPE_TO_KERNEL_TYPE.keys():
        for k in PYTYPE_TO_KERNEL_TYPE.keys():
            if k == restype.type:
                return k
    return restype.type
