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
import unittest

from matx.runtime import _ffi_api
from matx.runtime.object_generic import to_runtime_object


def make_ft_list(seq=()):
    new_seqs = [to_runtime_object(x) for x in seq]
    return _ffi_api.FTList(*new_seqs)


def make_list(seq=()):
    new_seqs = [to_runtime_object(x) for x in seq]
    return _ffi_api.List(*new_seqs)


class TestFTListFFI(unittest.TestCase):

    def test_get_item(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        item_0 = _ffi_api.ListGetItem(a, 0)
        item_1 = _ffi_api.ListGetItem(b, 0)
        print('{}, {}'.format(item_0, item_1))

    def test_get_slice(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        item_0 = _ffi_api.ListGetSlice(a, 0, 2, 1)
        item_1 = _ffi_api.ListGetSlice(b, 0, 2, 1)
        print('{}, {}'.format(item_0, item_1))

    def test_set_item(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        _ffi_api.ListSetItem(a, 0, 6)
        _ffi_api.ListSetItem(b, 0, 6)
        print('{}, {}'.format(a, b))

    def test_size(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        item_0 = _ffi_api.ListSize(a)
        item_1 = _ffi_api.ListSize(b)
        print('{}, {}'.format(item_0, item_1))

    def test_append(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        _ffi_api.ListAppend(a, 6)
        _ffi_api.ListAppend(b, 6)
        print('{}, {}'.format(a, b))

    def test_extend(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        _ffi_api.ListExtend(a, [6, 7])
        _ffi_api.ListExtend(b, [6, 7])
        print('{}, {}'.format(a, b))

    def test_repeat(self):
        a = make_ft_list([1, 2, 3])
        b = make_list([1, 2, 3])
        item_0 = _ffi_api.ListRepeat(a, 2)
        item_1 = _ffi_api.ListRepeat(b, 2)
        print('{}, {}'.format(item_0, item_1))

    def test_contains(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        item_0 = _ffi_api.ListContains(a, 2)
        item_1 = _ffi_api.ListContains(b, 2)
        print('{}, {}'.format(item_0, item_1))

    def test_set_slice(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        _ffi_api.ListSetSlice(a, 0, 2, [6, 7])
        _ffi_api.ListSetSlice(b, 0, 2, [6, 7])
        print('{}, {}'.format(a, b))

    def test_pop(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        item_0 = _ffi_api.ListPop(a)
        item_1 = _ffi_api.ListPop(b)
        print('{}, {}'.format(item_0, item_1))
        print('{}, {}'.format(a, b))
        item_0 = _ffi_api.ListPop(a, 0)
        item_1 = _ffi_api.ListPop(b, 0)
        print('{}, {}'.format(item_0, item_1))
        print('{}, {}'.format(a, b))

    def test_remove(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        _ffi_api.ListRemove(a, 3)
        _ffi_api.ListRemove(b, 3)
        print('{}, {}'.format(a, b))

    def test_reverse(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        _ffi_api.ListReverse(a)
        _ffi_api.ListReverse(b)
        print('{}, {}'.format(a, b))

    def test_clear(self):
        a = make_ft_list([1, 2, 3, 4, 5])
        b = make_list([1, 2, 3, 4, 5])
        _ffi_api.ListClear(a)
        _ffi_api.ListClear(b)
        print('{}, {}'.format(a, b))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
