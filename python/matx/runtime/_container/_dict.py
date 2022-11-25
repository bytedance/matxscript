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

from ... import _ffi
from .. import _ffi_api
from ..object import Object
from ..object_generic import to_runtime_object


@_ffi.register_object("FTDict")
@_ffi.register_object("runtime.Dict")
class Dict(Object):
    """matx.Dict implemented refering to python built-in dict,
       supports common methods of built-in list and some custom methods.

    Dict() -> construct empty dict
        >>> import matx
        >>> d = matx.Dict()
        >>> print(d)
        {}
    Dict(mapping) -> construct dict from mapping
        >>> import matx
        >>> d = matx.Dict({'a': 1, 'b':2})
        >>> print(d)
        {a: 1, b: 2}
    """
    __hash__ = None

    def __init__(self, seq=None, **kwargs):
        if seq:
            new_seqs = list()
            if isinstance(seq, dict):
                for k, v in seq.items():
                    new_seqs.append(to_runtime_object(k))
                    new_seqs.append(to_runtime_object(v))
            else:
                raise RuntimeError("[Dict.__init__] seq type is not support")
            self.__init_handle_by_constructor__(_ffi_api.Dict, *new_seqs)
        else:
            self.__init_handle_by_constructor__(_ffi_api.Dict)

    def __setstate__(self, state):
        assert isinstance(state, (bytes, bytearray))
        arr = _ffi_api.msgpack_loads(state)
        assert isinstance(arr, Dict), "internal error"
        handle, code = _ffi.matx_script_api.steal_object_handle(arr)
        self.handle = handle
        self.type_code = code

    def __getstate__(self):
        return _ffi_api.msgpack_dumps(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def __getitem__(self, k):
        """ x.__getitem__(y) <==> x[y] """
        return _ffi_api.DictGetItem(self, to_runtime_object(k))

    def __setitem__(self, key, value):
        """ Set self[key] to value. """
        return _ffi_api.DictSetItem(self, to_runtime_object(key), to_runtime_object(value))

    # def __delitem__(self, *args, **kwargs):
    #     """ Delete self[key]. """
    #     pass

    def __len__(self):
        return _ffi_api.DictSize(self)

    def __contains__(self, k):
        """ True if D has a key k, else False. """
        return _ffi_api.DictContains(self, k)

    # def __getattribute__(self, *args, **kwargs):
    #     """ Return getattr(self, name). """
    #     pass
    #
    # def __iter__(self, *args, **kwargs):
    #     """ Implement iter(self). """
    #     pass

    # def __sizeof__(self):
    #     """ D.__sizeof__() -> size of D in memory, in bytes """
    #     pass

    def __eq__(self, other):
        """ Return self==value. """
        return _ffi_api.DictEqual(self, other)

    def __ne__(self, other):
        """ Return self!=value. """
        return not self.__eq__(other)

    #
    # def __ge__(self, *args, **kwargs):
    #     """ Return self>=value. """
    #     pass
    #
    # def __gt__(self, *args, **kwargs):
    #     """ Return self>value. """
    #     pass
    #
    # def __le__(self, *args, **kwargs):
    #     """ Return self<=value. """
    #     pass
    #
    # def __lt__(self, *args, **kwargs):
    #     """ Return self<value. """
    #     pass

    def clear(self):
        """Remove all items.

        Returns:
            None

        Examples:
            >>> import matx
            >>> d = matx.Dict({'a': 1, 'b': 2})
            >>> d
            {a: 1, b: 2}
            >>> d.clear()
            >>> d
            {}
        """
        _ffi_api.DictClear(self)

    def reserve(self, new_size):
        """Increase the capacity of the dict to a value that's greater or equal to new_size.

        Args:
            new_size (int)

        Returns:
            None

        Examples:
            >>> d = matx.Dict({'a': 1, 'b': 2})
            >>> print(d.bucket_count())
            4
            >>> d.reserve(10)
            >>> print(d.bucket_count())
            32
        """
        _ffi_api.DictReserve(self, new_size)

    def bucket_count(self):
        """Returns the number of slots in the hash table.

        Returns:
            int

        Examples:
            >>> d = matx.Dict({'a': 1, 'b': 2})
            >>> print(d.bucket_count())
            4
        """
        return _ffi_api.DictBucketCount(self)

    # def copy(self):
    #     """ D.copy() -> a shallow copy of D """
    #     return _ffi_api.DictCopy(self)

    # @staticmethod  # known case
    # def fromkeys(*args, **kwargs):
    #     """ Returns a new dict with keys from iterable and values equal to value. """
    #     pass

    def get(self, k, d=None):
        """Return the value for key if key is in the dictionary, d.

        Args:
            k (item):
            d (item): defautl return value when k is not in dict

        Returns:
            item

        Examples:
            >>> import matx
            >>> d = matx.Dict({'a': 1, 'b': 2})
            >>> d.get('a')
            1
            >>> d.get('a', 3)
            1
            >>> d.get('c', 3)
            3
        """
        return _ffi_api.DictGetDefault(self, k, d)

    def items(self):
        """Return a key-value iterable (matx.Iterator).

        Returns:
            matx.Iterator

        Examples:
            >>> import matx
            >>> d = matx.Dict({'a': 1, 'b': 2})
            >>> it = d.items()
            >>> type(it)
            <class 'matx.runtime._container._iterator.Iterator'>
            >>> for k, v in it:
            ...     print(k, v)
            ...
            a 1
            b 2
            """
        return _ffi_api.Dict_ItemIter(self)

    def keys(self):
        """Return a key iterable.

        Returns:
            matx.Iterator

        Examples:
            >>> import matx
            >>> d = matx.Dict({'a': 1, 'b': 2})
            >>> it = d.keys()
            >>> type(it)
            <class 'matx.runtime._container._iterator.Iterator'>
            >>> for k in it:
            ...     print(k)
            ...
            a
            b
        """
        return _ffi_api.Dict_KeyIter(self)

    def pop(self, k, *args):
        """.pop(k[,d]) -> v, remove specified key and return the corresponding value.
           If key is not found, d is returned if given, otherwise Exception is raised

        Args:
            k(item)

        Returns:
            item

        Examples:
            >>> import matx
            >>> d = matx.Dict({'a': 1, 'b': 2, 'c': 3})
            >>> d.pop('a')
            1
            >>> d.pop('a', 100)
            100
            >>> d.pop('a')
            Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "/data00/shubaihan/repos/matx4/python/matx/runtime/_container/_dict.py", line 223, in pop
                return _ffi_api.DictPop(self, k, *args)
            TypeError: MATXError: dict.pop KeyError
            Stack trace:
            File "/data00/shubaihan/repos/matx4/src/runtime/container/dict_ref.cc", line 305
            [bt] (0) /data00/shubaihan/repos/matx4/lib/libmatx.so(+0x33c9ed) [0x7f70d4ad39ed]
            [bt] (1) /data00/shubaihan/repos/matx4/lib/libmatx.so(matx::runtime::Dict::pop(matx::runtime::PyArgs)+0x92) [0x7f70d4ad5362]
            [bt] (2) /data00/shubaihan/repos/matx4/lib/libmatx.so(+0x33f434) [0x7f70d4ad6434]
            [bt] (3) /data00/shubaihan/repos/matx4/lib/libmatx.so(MATXFuncCall_PYTHON_C_API+0x45) [0x7f70d4ab8c45]
        """
        return _ffi_api.DictPop(self, k, *args)

    # def popitem(self):
    #     """
    #     D.popitem() -> (k, v), remove and return some (key, value) pair as a
    #     2-tuple; but raise KeyError if D is empty.
    #     """
    #     pass
    #
    # def setdefault(self, k, d=None):
    #     """ D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D """
    #     pass
    #
    # def update(self, E=None, **F):  # known special case of dict.update
    #     """
    #     D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
    #     If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
    #     If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
    #     In either case, this is followed by: for k in F:  D[k] = F[k]
    #     """
    #     pass
    #

    def values(self):
        """Return a value iterable.

        Returns:
            matx.Iterator

        Examples:
            >>> import matx
            >>> d = matx.Dict({'a': 1, 'b': 2})
            >>> it = d.values()
            >>> type(it)
            <class 'matx.runtime._container._iterator.Iterator'>
            >>> for v in it:
            ...     print(v)
            ...
            1
            2
        """
        return _ffi_api.Dict_ValueIter(self)
