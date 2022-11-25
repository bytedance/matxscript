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


@_ffi.register_object("FTList")
@_ffi.register_object("List")
class List(Object):
    """matx.List implemented refering to python built-in list,
       supports common methods of built-in list and some custom methods.

    List() -> construct empty list
        >>> import matx
        >>> l = matx.List()
        >>> print(l)
        []
    List(iterable) -> construct list from iterable object
        >>> import matx
        >>> l = matx.List([1, 2, 3])
        >>> print(l)
        [1, 2, 3]
        >>> l1 = matx.List([1, 2, 3])
        >>> l2 = matx.List(l1)
        >>> print(l2)
        [1, 2, 3]
    """
    __hash__ = None

    def __init__(self, seq=()):
        new_seqs = [to_runtime_object(x) for x in seq]
        self.__init_handle_by_constructor__(_ffi_api.List, *new_seqs)

    def __setstate__(self, state):
        assert isinstance(state, (bytes, bytearray))
        arr = _ffi_api.msgpack_loads(state)
        assert isinstance(arr, List), "internal error"
        handle, code = _ffi.matx_script_api.steal_object_handle(arr)
        self.handle = handle
        self.type_code = code

    def __getstate__(self):
        return _ffi_api.msgpack_dumps(self)

    def __repr__(self):
        return _ffi_api.RTValue_Repr(self)

    def __iter__(self):
        return _ffi_api.List_Iter(self)

    def __getitem__(self, idx):
        length = len(self)
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else length
            step = idx.step if idx.step is not None else 1
            return _ffi_api.ListGetSlice(self, start, stop, step)
        else:
            if idx < 0:
                idx += length
            if idx < -length or idx >= length:
                raise IndexError("Index out of range. size: {}, got index {}"
                                 .format(length, idx))
            return _ffi_api.ListGetItem(self, idx)

    def __setitem__(self, key, value):
        """ Set self[key] to value. """
        return _ffi_api.ListSetItem(self, key, value)

    def __delitem__(self, key):
        """ Delete self[key]. """
        return _ffi_api.ListDelItem(self, key)

    def __len__(self):
        return _ffi_api.ListSize(self)

    def append(self, p_object):
        """Append object to the end of the list.

        Args:
            p_object

        Returns:
            None

        Examples:
            >>> import matx
            >>> l = matx.List()
            >>> l.append(1)
            >>> l
            [1]
        """
        return _ffi_api.ListAppend(self, to_runtime_object(p_object))

    def __add__(self, value):
        """ Return self+value. """
        assert isinstance(value, (type(self), list))
        ret = List.__new__(List)
        ret.__init_handle_by_constructor__(_ffi_api.ListConcat, self, value)
        return ret

    def __iadd__(self, value):
        """ Implement self+=value. """
        assert isinstance(value, (type(self), list))
        _ffi_api.ListExtend(self, value)
        return self

    def __mul__(self, *args, **kwargs):
        """ Return self*value. """
        assert len(args) == 1
        times = args[0]
        assert isinstance(times, int)
        return _ffi_api.ListRepeat(self, times)

    #
    # def __imul__(self, *args, **kwargs):
    #     """ Implement self*=value. """
    #     return self._data.__imul__(*args, **kwargs)

    def __contains__(self, item):
        """ Return key in self. """
        return _ffi_api.ListContains(self, item)

    def __eq__(self, other):
        """ Return self==value. """
        return _ffi_api.ListEqual(self, other)

    def __ne__(self, other):
        """ Return self!=value. """
        return not self.__eq__(other)

    # def __ge__(self, *args, **kwargs):
    #     """ Return self>=value. """
    #     pass
    #
    # def __gt__(self, *args, **kwargs):
    #     """ Return self>value. """
    #     pass

    # def __le__(self, *args, **kwargs):
    #     """ Return self<=value. """
    #     pass
    #
    # def __lt__(self, *args, **kwargs):
    #     """ Return self<value. """
    #     pass

    # def __getattribute__(self, *args, **kwargs):
    #     """ Return getattr(self, name). """
    #     pass

    # def __iter__(self, *args, **kwargs):
    #     """ Implement iter(self). """
    #     pass

    # @staticmethod  # known case of __new__
    # def __new__(*args, **kwargs):
    #     """ Create and return a new object.  See help(type) for accurate signature. """
    #     pass

    # def __reversed__(self):
    #     """ L.__reversed__() -- return a reverse iterator over the list """
    #     pass

    # def __rmul__(self, *args, **kwargs):
    #     """ Return value*self. """
    #     pass

    # def __sizeof__(self):
    #     """ L.__sizeof__() -- size of L in memory, in bytes """
    #     pass

    def clear(self):
        """Remove all items from list.

        Returns:
            None

        Examples:
            >>> import matx
            >>> l = matx.List([1, 2, 3])
            >>> l
            [1, 2, 3]
            >>> l.clear()
            >>> l
            []
        """
        _ffi_api.ListClear(self)

    # def copy(self):
    #     """ L.copy() -> list -- a shallow copy of L """
    #     return List(self._data.copy())

    # def count(self, value):
    #     """ L.count(value) -> integer -- return number of occurrences of value """
    #     return self._data.count(value)

    def extend(self, iterable):
        """Extend list by appending elements from the iterable.

        Args:
            iterable : iterable

        Returns:
            None

        Examples:
            >>> import matx
            >>> l = matx.List([1, 2, 3])
            >>> l.extend(matx.List([1, 2, 3]))
            >>> print(l)
            [1, 2, 3, 1, 2, 3]
            >>> l.extend([1, 2, 3])
            >>> print(l)
            [1, 2, 3, 1, 2, 3, 1, 2, 3]
        """
        return _ffi_api.ListExtend(self, iterable)

    def reserve(self, new_size):
        """Increase the capacity of the list to a value that's greater or equal to new_size.

        Args:
            new_size (int)

        Returns:
            None

        Examples:
            >>> import matx
            >>> l = matx.List([1, 2, 3])
            >>> print(l)
            [1, 2, 3]
            >>> print(l.capacity())
            4
            >>> l.reserve(8)
            >>> print(l)
            [1, 2, 3]
            >>> print(l.capacity())
            8
        """
        return _ffi_api.ListReserve(self, new_size)

    def capacity(self):
        """Return the number of elements that the list has currently allocated space for.

        Returns:
            int

        Examples:
            >>> import matx
            >>> l = matx.List([1, 2, 3])
            >>> print(l)
            [1, 2, 3]
            >>> print(l.capacity())
            4
        """
        return _ffi_api.ListCapacity(self)

    # def index(self, value, start=None, stop=None):
    #     """
    #     L.index(value, [start, [stop]]) -> integer -- return first index of value.
    #     Raises ValueError if the value is not present.
    #     """
    #     return self._data.index(value, start, stop)

    # def insert(self, index, p_object):
    #     """ L.insert(index, object) -- insert object before index """
    #     return self._data.insert(index, p_object)

    def pop(self, index=-1):
        """Remove and return item at index (default last).
           Raises Exception if list is empty or index is out of range.

        Args:
            index (int, optional)

        Returns:
            item

        Raises:
            Exception

        Examples:
            >>> import matx
            >>> l = matx.List([1, 2, 3])
            >>> l.pop()
            3
            >>> import matx
            >>> l = matx.List([1, 2, 3, 4, 5])
            >>> l.pop()
            5
            >>> l
            [1, 2, 3, 4]
            >>> l.pop(1)
            2
            >>> l
            [1, 3, 4]
        """
        return _ffi_api.ListPop(self, index)

    def insert(self, index, value):
        """Insert an item at a given position. The first argument is the index of the element before which to insert

        Args:
            index (int)
            value (value type)

        Returns:
            None

        """
        return _ffi_api.ListInsert(self, index, value)

    def index(self, x, start=None, end=None):
        """Return zero-based index in the list of the first item whose value is equal to x.

        Args:
            x (value type)
            start (int)
            end (int)

        Raises:
            a ValueError if there is no such item.

        Returns:
            None

        """
        if start is None:
            start = 0
        if end is None:
            end = len(self)
        return _ffi_api.ListIndex(self, x, start, end)

    def remove(self, value):
        """Remove first occurrence of value.
           Raises Exception if the value is not present.


        Args:
            value

        Returns:
            None

        Raises:
            Exception

        Examples:
            >>> import matx
            >>> l = matx.List(["a", "b", "c", "d", "c"])
            >>> l.remove("c")
            >>> l
            [a, b, d, c]
        """
        return _ffi_api.ListRemove(self, value)

    def reverse(self):
        """Reverse *IN PLACE*.

        Returns:
            None

        Examples:
            >>> imprt matx
            >>> l = matx.List([1, 2, 3, 4])
            >>> l.reverse()
            >>> l
            [4, 3, 2, 1]
        """
        return _ffi_api.ListReverse(self)

    # def sort(self, key=None, reverse=False):
    #     """ L.sort(key=None, reverse=False) -> None -- stable sort *IN PLACE* """
    #     return COWList(self._data.sort(key=key, reverse=reverse))


def default_comp(x, y):
    if x < y:
        return -1
    if x > y:
        return 1
    return 0


def shiftdown(l, i, comp=None):
    assert isinstance(l, (list, List))
    length = len(l)
    if not comp:
        comp = default_comp

    while True:
        left_i = 2 * i + 1
        right_i = left_i + 1
        min_index = i
        if left_i < length and comp(l[left_i], l[min_index]) < 0:
            min_index = left_i
        if right_i < length and comp(l[right_i], l[min_index]) < 0:
            min_index = right_i
        if min_index == i:
            break
        l[i], l[min_index] = l[min_index], l[i]
        i = min_index


def heapify(l, comp=None):
    assert isinstance(l, (list, List))
    length = len(l)
    if not comp:
        comp = default_comp
    if length <= 1:
        return
    last_index = (length - 2) // 2
    for i in reversed(range(last_index + 1)):
        shiftdown(l, i, comp)


def heap_replace(l, item, comp=None):
    l[0] = item
    shiftdown(l, 0, comp)


def heap_pushpop(l, item, comp=None):
    if not comp:
        comp = default_comp
    if comp(item, l[0]) > 0:
        ret, l[0] = l[0], item
        shiftdown(l, 0, comp)
        return ret
    else:
        return item


def _partition(l, s_pos, e_pos, t_pos, comp):
    if t_pos == e_pos:
        # find max
        max_pos = e_pos
        for i in reversed(range(s_pos, e_pos)):
            if comp(l[i], l[max_pos]) > 0:
                max_pos = i
        if max_pos != e_pos:
            l[max_pos], l[e_pos] = l[e_pos], l[max_pos]
        return
    if t_pos == s_pos:
        # find min
        min_pos = s_pos
        for i in range(s_pos + 1, e_pos + 1):
            if comp(l[i], l[min_pos]) < 0:
                min_pos = i
        if min_pos != s_pos:
            l[min_pos], l[s_pos] = l[s_pos], l[min_pos]
        return

    i, j = s_pos, e_pos
    v = l[e_pos]
    while i < j:
        while i < j:
            if comp(l[i], v) > 0:
                l[j] = l[i]
                j -= 1
                break
            else:
                i += 1
        while i < j:
            if comp(l[j], v) < 0:
                l[i] = l[j]
                i += 1
                break
            else:
                j -= 1
    l[i] = v
    if i == t_pos:
        return
    elif i > t_pos:
        _partition(l, s_pos, i - 1, t_pos, comp)
    else:
        _partition(l, i + 1, e_pos, t_pos, comp)


def nth_element(l, n, comp=None):
    assert isinstance(l, (list, List))
    length = len(l)
    if length < n or n < 1:
        return
    if not comp:
        comp = default_comp
    _partition(l, 0, length - 1, n - 1, comp)
