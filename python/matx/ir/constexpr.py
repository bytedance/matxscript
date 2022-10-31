# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from . import _ffi_api


def _scalar_type_inference(value):
    if hasattr(value, 'dtype'):
        dtype = str(value.dtype)
    elif isinstance(value, bool):
        dtype = 'bool'
    elif isinstance(value, float):
        dtype = 'float64'
    elif isinstance(value, int):
        dtype = 'int64'
    else:
        raise NotImplementedError('Cannot automatically inference the type.'
                                  ' value={}'.format(value))
    return dtype


def const(value, dtype=None):
    """construct a constant

    Parameters
    ----------
    value : Union[int, float, bool]
        The content of the constant number.

    dtype : str or None, optional
        The data type.

    Returns
    -------
    const_val: Expr
        The result expression.
    """
    if dtype is None:
        dtype = _scalar_type_inference(value)
    if dtype == "uint64" and value >= (1 << 63):
        return _ffi_api.LargeUIntImm(
            dtype, value & ((1 << 32) - 1), value >> 32)
    return _ffi_api._const(value, dtype)


def generic_const(value):
    """build const expr"""
    from .expr import UnicodeImm, StringImm, NoneExpr
    if value is None:
        return NoneExpr()
    elif isinstance(value, bool):
        return const(value, "bool")
    elif isinstance(value, float):
        return const(value, "float64")
    elif isinstance(value, int):
        return const(value, "int64")
    elif isinstance(value, str):
        return UnicodeImm(value)
    elif isinstance(value, bytes):
        return StringImm(value)
    else:
        raise NotImplementedError('Cannot automatically convert the const value.'
                                  ' value={}'.format(value))
