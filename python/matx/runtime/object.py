# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the Object is inspired by incubator-tvm.
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
# pylint: disable=invalid-name, unused-import
"""Runtime Object API"""
import ctypes

from .._ffi.base import _RUNTIME_ONLY, check_call, _LIB, c_str
from .._ffi.runtime_ctypes import ObjectRValueRef
from . import _ffi_api

# pylint: disable=wrong-import-position,unused-import
from .._ffi._selector import _set_class_object, _set_class_object_generic
from .._ffi._selector import ObjectBase


def _new_object(cls):
    """Helper function for pickle"""
    return cls.__new__(cls)


class Object(ObjectBase):
    """Base class for all tvm's runtime objects."""
    __slots__ = []

    def __repr__(self):
        object_text = _ffi_api.AsRepr(self)
        if isinstance(object_text, (bytes, bytearray)):
            object_text = object_text.decode('utf-8')
        return object_text

    def __dir__(self):
        class_names = dir(self.__class__)
        attr_names = _ffi_api.NodeListAttrNames(self)
        size = len(attr_names)
        return sorted([attr_names[i].decode() for i in range(size)] + class_names)

    def __getattr__(self, name):
        if name == "__init_self__" or name == "__call__" or name in self.__slots__:
            raise AttributeError(f"{name} is not set")

        try:
            success, attr = _ffi_api.NodeGetAttr(self, name)
            if success:
                return attr
            else:
                raise AttributeError(
                    "%s has no attribute %s" % (str(type(self)), name)) from None
        except:
            raise AttributeError(
                "%s has no attribute %s" % (str(type(self)), name)) from None

    def __hash__(self):
        return _ffi_api.ObjectPtrHash(self)

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls,), self.__getstate__())

    def _move(self):
        """Create an RValue reference to the object and mark the object as moved.

        This is a advanced developer API that can be useful when passing an
        unique reference to an Object that you no longer needed to a function.

        A unique reference can trigger copy on write optimization that avoids
        copy when we transform an object.

        Note
        ----
        All the reference of the object becomes invalid after it is moved.
        Be very careful when using this feature.

        Examples
        --------

        .. code-block:: python

           x = matx.ir.PrimVar("x", "int32")
           x0 = x
           some_packed_func(x._move())
           # both x0 and x will points to None after the function call.

        Returns
        -------
        rvalue : The rvalue reference.
        """
        return ObjectRValueRef(self)


_set_class_object(Object)
