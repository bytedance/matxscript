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


@_ffi.register_object("FTSet")
@_ffi.register_object("runtime.Set")
class Set(Object):
    """matx.Set: matx.Set implemented refering to python built-in dict,
       supports common methods of built-in list and some custom methods.

    set() -> construct empty set
    Examples:
        >>> import matx
        >>> s = matx.Set()
        >>> print(s)
        {}

    set(iterable) -> construct set from iterable
    Examples:
        >>> import matx
        >>> s = matx.Set(['a', 1, 'b'])
        >>> print(s)
        {a, b, 1}
    """

    __hash__ = None

    def __init__(self, seq=()):
        new_seqs = [to_runtime_object(x) for x in seq]
        self.__init_handle_by_constructor__(_ffi_api.Set, *new_seqs)

    def __setstate__(self, state):
        assert isinstance(state, (bytes, bytearray))
        arr = _ffi_api.msgpack_loads(state)
        assert isinstance(arr, Set), "internal error"
        handle, code = _ffi.matx_script_api.steal_object_handle(arr)
        self.handle = handle
        self.type_code = code

    def __getstate__(self):
        return _ffi_api.msgpack_dumps(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def __iter__(self):
        return _ffi_api.Set_Iter(self)

    def __contains__(self, key):
        """ x.__contains__(y) <==> y in x. """
        return _ffi_api.SetContains(self, to_runtime_object(key))

    def __len__(self):
        """ Return len(self). """
        return _ffi_api.SetSize(self)

    def add(self, p_object):
        """Add an element to a set.
           This has no effect if the element is already present.

        Returns
            None

        Examples:
            >>> import matx
            >>> s = matx.Set(['a', 1, 'b'])
            >>> s.add('c')
            >>> print(s)
            {a, c, b, 1}
        """
        return _ffi_api.SetAddItem(self, to_runtime_object(p_object))

    def clear(self):
        """Remove all elements.

        Returns:
            None

        Examples:
            >>> import matx
            >>> s = matx.Set(['a', 1, 'b'])
            >>> print(s)
            {a, b, 1}
            >>> s.clear()
            >>> print(s)
            {}
        """
        return _ffi_api.SetClear(self)

    def reserve(self, new_size):
        """Increase the capacity of the set to a value that's greater or equal to new_size.

        Args:
            new_size (int)

        Returns:
            None

        Examples:
            >>> import matx
            >>> s = matx.Set(['a', 1, 'b'])
            >>> print(s.bucket_count())
            8
            >>> s.reserve(10)
            >>> print(s.bucket_count())
            32
        """
        _ffi_api.SetReserve(self, new_size)

    def bucket_count(self):
        """Returns the number of slots in the hash table.

        Returns:
            int

        Examples:
            >>> import matx
            >>> s = matx.Set(['a', 1, 'b'])
            >>> print(s.bucket_count())
            8
        """
        return _ffi_api.SetBucketCount(self)

    # def copy(self, *args, **kwargs):
    #     """ Return a shallow copy of a set. """
    #     return _ffi_api.SetCopy(self)

    def update(self, *args, **kwargs):
        """Update a set with the union of itself and others.

        Returns:
            None

        Examples:
            >>> import matx
            >>> a = matx.Set({'a', 'b', 'c'})
            >>> a.update(matx.Set({'b', 'd'}), {'e', 'f'}, ['a', 1], matx.List([1, 2]))
            >>> a
            {'a', 'c', 'd', 2, 'b', 'f', 1, 'e'}
        """
        return _ffi_api.SetUpdate(self, *args)

    # def __getattribute__(self, *args, **kwargs):
    #     """ Return getattr(self, name). """
    #     pass

    # def __and__(self, *args, **kwargs):
    #     """ Return self&value. """
    #     pass
    #
    # def __iand__(self, *args, **kwargs):
    #     """ Return self&=value. """
    #     pass
    #
    # def __or__(self, *args, **kwargs):
    #     """ Return self|value. """
    #     pass
    #
    # def __ior__(self, *args, **kwargs):
    #     """ Return self|=value. """
    #     pass
    #
    # def __ixor__(self, *args, **kwargs):
    #     """ Return self^=value. """
    #     pass
    #
    # def __sub__(self, *args, **kwargs):
    #     """ Return self-value. """
    #     pass
    #
    # def __isub__(self, *args, **kwargs):
    #     """ Return self-=value. """
    #     pass
    #
    # def __ror__(self, *args, **kwargs):
    #     """ Return value|self. """
    #     pass
    #
    # def __rsub__(self, *args, **kwargs):
    #     """ Return value-self. """
    #     pass
    #
    # def __xor__(self, *args, **kwargs):
    #     """ Return self^value. """
    #     pass
    #
    # def __rxor__(self, *args, **kwargs):
    #     """ Return value^self. """
    #     pass
    #
    # def __iter__(self, *args, **kwargs):
    #     """ Implement iter(self). """
    #     pass

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

    def __eq__(self, other):
        """ Return self==value. """
        return _ffi_api.SetEqual(self, other)

    def __ne__(self, other):
        """ Return self!=value. """
        return not self.__eq__(other)

    # def __rand__(self, *args, **kwargs):
    #     """ Return value&self. """
    #     pass

    # def __reduce__(self, *args, **kwargs):
    #     """ Return state information for pickling. """
    #     pass
    #
    # def __sizeof__(self):
    #     """ S.__sizeof__() -> size of S in memory, in bytes """
    #     pass

    def difference(self, *args):
        """Return the difference of two or more sets as a new set.

        Returns:
            matx.Set

        Examples:
            >>> import matx
            >>> s = matx.Set({'a', 'b', 'c', 'd', 'e', 'f'})
            >>> s.difference(matx.Set({'a', 'b'}), {'d', 1}, ['f'], matx.List([2]))
            {'c', 'e'}
            >>> s
            {'a', 'd', 'c', 'b', 'f', 'e'}
        """
        return _ffi_api.SetDifference(self, *args)

    def difference_update(self, *args):
        """Remove all elements of another set from this set.

        Returns:
            None

        Examples:
            >>> s = matx.Set({'a', 'b', 'c', 'd', 'e', 'f'})
            >>> s.difference_update(matx.Set({'a', 'b'}), {'d', 1}, ['f'], matx.List([2]))
            >>> s
            {'c', 'e'}
        """
        return _ffi_api.SetDifferenceUpdate(self, *args)

    def discard(self, item):
        """Remove an element from a set if it is a member.
           If the element is not a member, do nothing.

        Returns:
            None

        Examples:
            >>> import matx
            >>> s = matx.Set({'a', 'b', 'c'})
            >>> s.discard('a')
            >>> s
            {'c', 'b'}
            >>> s.discard('d')
            >>> s
            {'c', 'b'}
        """
        return _ffi_api.SetDiscard(self, item)

    # def intersection(self, *args, **kwargs):
    #     """
    #     Return the intersection of two sets as a new set.
    #
    #     (i.e. all elements that are in both sets.)
    #     """
    #     pass
    #
    # def intersection_update(self, *args, **kwargs):
    #     """ Update a set with the intersection of itself and another. """
    #     pass
    #
    # def isdisjoint(self, *args, **kwargs):
    #     """ Return True if two sets have a null intersection. """
    #     pass
    #
    # def issubset(self, *args, **kwargs):
    #     """ Report whether another set contains this set. """
    #     pass
    #
    # def issuperset(self, *args, **kwargs):
    #     """ Report whether this set contains another set. """
    #     pass

    # def pop(self, *args, **kwargs):
    #     """
    #     Remove and return an arbitrary set element.
    #     Raises KeyError if the set is empty.
    #     """
    #     pass
    #
    # def remove(self, *args, **kwargs):
    #     """
    #     Remove an element from a set; it must be a member.
    #
    #     If the element is not a member, raise a KeyError.
    #     """
    #     pass
    #
    # def symmetric_difference(self, *args, **kwargs):
    #     """
    #     Return the symmetric difference of two sets as a new set.
    #
    #     (i.e. all elements that are in exactly one of the sets.)
    #     """
    #     pass
    #
    # def symmetric_difference_update(self, *args, **kwargs):
    #     """ Update a set with the symmetric difference of itself and another. """
    #     pass
    #

    def union(self, *args, **kwargs):
        """Return the union of sets as a new set.

        Returns:
            matx.Set

        Examples:
            >>> import matx
            >>> a = matx.Set({'a', 'b', 'c'})
            >>> a.union(matx.Set({'b', 'd'}), {'e', 'f'}, ['a', 1], matx.List([1, 2]))
            {'a', 'c', 'd', 2, 'b', 'f', 1, 'e'}
            >>> a
            {'a', 'c', 'b'}
        """
        return _ffi_api.SetUnion(self, *args)
