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
"""Runtime container structures."""
from .. import _ffi
from .object import Object
from .object_generic import ObjectTypes, to_runtime_object
from . import _ffi_api
from . import Object

from ._container import Dict, List, Set
from ._container import Iterator
from ._container import Int32Generator
from ._container import Int64Generator
from ._container import Float32Generator
from ._container import Float64Generator
from ._container import BoolGenerator
from ._container import RTValueGenerator
from ._container import OpaqueObject
from ._container import UserData


def slice_index_correction(index, length):
    if index < 0:
        index += length
        if index < 0:
            index = 0
    else:
        if index > length:
            index = length
    return index


def getitem_helper(obj, elem_getter, length, idx):
    """Helper function to implement a pythonic getitem function.

    Parameters
    ----------
    obj: object
        The original object

    elem_getter : function
        A simple function that takes index and return a single element.

    length : int
        The size of the array

    idx : int or slice
        The argument passed to getitem

    Returns
    -------
    result : object
        The result of getitem
    """
    if isinstance(idx, slice):
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else length
        step = idx.step if idx.step is not None else 1
        start = slice_index_correction(start, length)
        stop = slice_index_correction(stop, length)
        return [elem_getter(obj, i) for i in range(start, stop, step)]

    if idx < -length or idx >= length:
        raise IndexError("Index out of range. size: {}, got index {}"
                         .format(length, idx))
    if idx < 0:
        idx += length
    return elem_getter(obj, idx)


@_ffi.register_object("runtime.Tuple")
class Tuple(Object):
    """tuple object.

    Parameters
    ----------
    fields : List[Object]
        The source.
    """

    def __init__(self, *fields):
        new_fields = [to_runtime_object(x) for x in fields]
        self.__init_handle_by_constructor__(_ffi_api.Tuple, *new_fields)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def __getitem__(self, idx):
        return getitem_helper(
            self, _ffi_api.GetTupleFields, len(self), idx)

    def __len__(self):
        return _ffi_api.GetTupleSize(self)

    def __eq__(self, other):
        """ Return self==value. """
        return _ffi_api.TupleEqual(self, other)


class ArrayIterator(object):

    def __init__(self, obj):
        self._obj = obj
        self._index = 0
        self._len = len(self._obj)

    def __next__(self):
        if self._index < self._len:
            item = self._obj[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration()

    def __iter__(self):
        return self


@_ffi.register_object("Array")
class Array(Object):
    """Array container of MATX.

    You do not need to create Array explicitly.
    Normally python list and tuple will be converted automatically
    to Array during tvm function call.
    You may get Array in return values of TVM function call.
    """

    def __init__(self, seq=()):
        for f in seq:
            object_exts = (str, bytes, bytearray) + ObjectTypes
            assert isinstance(f, object_exts), "Expect object, but received : {0}".format(type(f))
        self.__init_handle_by_constructor__(_ffi_api.Array, *seq)

    def __dir__(self):
        attrs = sorted([repr(i) for i in range(len(self))])
        return attrs

    def __getattr__(self, name):
        return self.__getitem__(int(name))

    def __getitem__(self, idx):
        return getitem_helper(
            self, _ffi_api.ArrayGetItem, len(self), idx)

    def __len__(self):
        return _ffi_api.ArraySize(self)

    def __iter__(self):
        return ArrayIterator(self)

    def __contains__(self, k):
        return _ffi_api.ArrayContains(self, k)


class MapIterator(object):

    def __init__(self, obj):
        self._items = obj.items()
        self._index = 0
        self._len = len(self._items)

    def __next__(self):
        if self._index < self._len:
            item = self._items[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration()

    def __iter__(self):
        return self


@_ffi.register_object
class Map(Object):
    """Map container of MATX IR.

    You do not need to create Map explicitly.
    Normally python dict will be converted automaticall to Map during tvm function call.
    You can use convert to create a dict[Object-> Object] into a Map
    """

    def __init__(self, seq=()):
        for f in seq:
            object_exts = (str, bytes, bytearray) + ObjectTypes
            assert isinstance(f, object_exts), "Expect object, but received : {0}".format(type(f))
        self.__init_handle_by_constructor__(_ffi_api.Map, *seq)

    def __getitem__(self, k):
        return _ffi_api.MapGetItem(self, k)

    def __contains__(self, k):
        return _ffi_api.MapCount(self, k) != 0

    def items(self):
        """Get the items from the map"""
        items = _ffi_api.MapItems(self)
        return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]

    def keys(self):
        """Get the keys from the map"""
        keys = _ffi_api.MapKeys(self)
        return [k for k in keys]

    def values(self):
        """Get the values from the map"""
        values = _ffi_api.MapValues(self)
        return [v for v in values]

    def __len__(self):
        return _ffi_api.MapSize(self)

    def __iter__(self):
        return MapIterator(self)

    def get(self, key, default=None):
        """Get an element with a default value.

        Parameters
        ----------
        key : object
            The attribute key.

        default : object
            The default object.

        Returns
        -------
        value: object
            The result value.
        """
        return self[key] if key in self else default
