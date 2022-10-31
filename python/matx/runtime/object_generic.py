# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the ObjectGeneric is inspired by incubator-tvm.
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
"""Common implementation of object generic related logic"""
# pylint: disable=unused-import, invalid-name
import ctypes
from numbers import Number, Integral
from .._ffi.base import string_types
from .._ffi.runtime_ctypes import ObjectRValueRef
from .._ffi._selector import _to_runtime_object

from . import _ffi_api
from .object import ObjectBase, _set_class_object_generic
from .module import Module


class ObjectGeneric(object):
    """Base class for all classes that can be converted to object."""

    def asobject(self):
        """Convert value to object"""
        raise NotImplementedError()


ObjectTypes = (ObjectBase, Module, ObjectRValueRef)


def _assert_is_object(p_object):
    msg = "Expect object, but received : {0}".format(type(p_object))
    assert isinstance(p_object, ObjectTypes), msg


_DirectReturnTypes = (Number, str, bytes, bytearray, bool) + \
    ObjectTypes + (ObjectGeneric, ctypes.c_void_p)


def to_runtime_object(value):
    if value is None or isinstance(value, _DirectReturnTypes):
        return value

    try:
        return _to_runtime_object(value)
    except BaseException as e:
        raise ValueError(
            "don't know how to convert type %s to matx runtime: reason: %s" % (type(value), e)
        ) from None


_set_class_object_generic(ObjectGeneric, to_runtime_object)
