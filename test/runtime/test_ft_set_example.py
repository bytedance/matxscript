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


def make_set(seq=(), is_ft=False):
    new_seqs = [to_runtime_object(x) for x in seq]
    if is_ft:
        return _ffi_api.FTSet(*new_seqs)
    else:
        return _ffi_api.Set(*new_seqs)


class TestFTSetFFI(unittest.TestCase):

    def test_construct(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        print('{}, {}'.format(a, b))

    def test_equal(self):
        a0 = make_set([1, 2, 3, 4, 5])
        a1 = make_set([1, 2, 3, 4, 5])
        b0 = make_set([1, 2, 3, 4, 5], is_ft=True)
        b1 = make_set([1, 2, 3, 4, 5], is_ft=True)
        ret_a = _ffi_api.SetEqual(a0, a1)
        ret_b = _ffi_api.SetEqual(b0, b1)
        print('{}, {}'.format(ret_a, ret_b))

    def test_iter(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        ret_a = _ffi_api.Set_Iter(a)
        ret_b = _ffi_api.Set_Iter(b)

    def test_size(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        ret_a = _ffi_api.SetSize(a)
        ret_b = _ffi_api.SetSize(b)
        print('{}, {}'.format(ret_a, ret_b))

    def test_contains(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        ret_a = _ffi_api.SetContains(a, 3)
        ret_b = _ffi_api.SetContains(b, 3)
        print('{}, {}'.format(ret_a, ret_b))

    def test_add(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        _ffi_api.SetAddItem(a, 6)
        _ffi_api.SetAddItem(b, 6)
        print('{}, {}'.format(a, b))

    def test_discard(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        _ffi_api.SetDiscard(a, 1)
        _ffi_api.SetDiscard(b, 1)
        print('{}, {}'.format(a, b))

    def test_clear(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        _ffi_api.SetClear(a)
        _ffi_api.SetClear(b)
        print('{}, {}'.format(a, b))

    def test_reserve(self):
        a = make_set([1, 2, 3, 4, 5])
        b = make_set([1, 2, 3, 4, 5], is_ft=True)
        _ffi_api.SetReserve(a, 6)
        _ffi_api.SetReserve(b, 6)
        print('{}, {}'.format(_ffi_api.SetBucketCount(a), _ffi_api.SetBucketCount(b)))

    def test_difference(self):
        a0 = make_set([1, 2, 3, 4, 5])
        a1 = make_set([2, 3, 6, 7])
        a2 = make_set([3, 4, 8, 9])
        b0 = make_set([1, 2, 3, 4, 5], is_ft=True)
        b1 = make_set([2, 3, 6, 7], is_ft=True)
        b2 = make_set([3, 4, 8, 9], is_ft=True)
        ret_a = _ffi_api.SetDifference(a0, a1, a2)
        ret_b = _ffi_api.SetDifference(b0, b1, b2)
        print('{}, {}'.format(ret_a, ret_b))

    def test_difference_update(self):
        a0 = make_set([1, 2, 3, 4, 5])
        a1 = make_set([2, 3, 6, 7])
        a2 = make_set([3, 4, 8, 9])
        b0 = make_set([1, 2, 3, 4, 5], is_ft=True)
        b1 = make_set([2, 3, 6, 7], is_ft=True)
        b2 = make_set([3, 4, 8, 9], is_ft=True)
        _ffi_api.SetDifferenceUpdate(a0, a1, a2)
        _ffi_api.SetDifferenceUpdate(b0, b1, b2)
        print('{}, {}'.format(a0, b0))

    def test_update(self):
        a0 = make_set([1, 2, 3, 4, 5])
        a1 = make_set([2, 3, 6, 7])
        a2 = make_set([3, 4, 8, 9])
        b0 = make_set([1, 2, 3, 4, 5], is_ft=True)
        b1 = make_set([2, 3, 6, 7], is_ft=True)
        b2 = make_set([3, 4, 8, 9], is_ft=True)
        _ffi_api.SetUpdate(a0, a1, a2)
        _ffi_api.SetUpdate(b0, b1, b2)
        print('{}, {}'.format(a0, b0))

    def test_union(self):
        a0 = make_set([1, 2, 3, 4, 5])
        a1 = make_set([2, 3, 6, 7])
        a2 = make_set([3, 4, 8, 9])
        b0 = make_set([1, 2, 3, 4, 5], is_ft=True)
        b1 = make_set([2, 3, 6, 7], is_ft=True)
        b2 = make_set([3, 4, 8, 9], is_ft=True)
        ret_a = _ffi_api.SetUnion(a0, a1, a2)
        ret_b = _ffi_api.SetUnion(b0, b1, b2)
        print('{}, {}'.format(ret_a, ret_b))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
