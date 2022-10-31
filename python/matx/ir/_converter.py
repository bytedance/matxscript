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

from numbers import Number
from .._ffi.base import string_types
from .._ffi import to_packed_func
from .. import runtime
from ..runtime.object import ObjectBase
from ..runtime.packed_func import PackedFuncBase
from ..runtime.object_generic import ObjectGeneric, ObjectTypes
from .constexpr import const


def to_ir_object(value):
    """Convert a Python value to corresponding object type.

    Parameters
    ----------
    value : Any
        The value to be inspected.

    Returns
    -------
    obj : Object
        The corresponding object value.
    """
    if isinstance(value, ObjectTypes):
        return value
    if isinstance(value, bool):
        return const(value, 'uint1x1')
    if isinstance(value, Number):
        return const(value)
    if isinstance(value, (bytes, bytearray)):
        return value
    if isinstance(value, string_types):
        return value.encode("utf-8")
    if isinstance(value, (list, tuple)):
        value = [to_ir_object(x) for x in value]
        return runtime.Array(value)
    if isinstance(value, dict):
        vlist = []
        for item in value.items():
            if (not isinstance(item[0], ObjectTypes) and
                    not isinstance(item[0], string_types)):
                raise ValueError("key of map must already been a container type")
            vlist.append(item[0])
            vlist.append(to_ir_object(item[1]))
        return runtime.Map(vlist)
    if isinstance(value, ObjectGeneric):
        return value.asobject()
    if value is None:
        return None

    raise ValueError("don't know how to convert type %s to object" % type(value))


def convert(value):
    """Convert value to MATX IR Object or function.

    Parameters
    ----------
    value : Any

    Returns
    -------
    ir_val : Object or Function
        Converted value in MATX
    """
    if isinstance(value, (PackedFuncBase, ObjectBase)):
        return value

    if callable(value):
        return to_packed_func(value)

    return to_ir_object(value)
