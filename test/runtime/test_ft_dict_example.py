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

from typing import Dict
from matx.runtime import _ffi_api
from matx.runtime.object_generic import to_runtime_object


def make_dict(seq: Dict, is_ft=False):
    new_seqs = []
    for k, v in seq.items():
        new_seqs.append(to_runtime_object(k))
        new_seqs.append(to_runtime_object(v))
    if is_ft:
        return _ffi_api.FTDict(*new_seqs)
    else:
        return _ffi_api.Dict(*new_seqs)


class TestFTDictFFI(unittest.TestCase):

    def test_construct(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        print('{}, {}'.format(a, b))

    def test_size(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.DictSize(a)
        ret_b = _ffi_api.DictSize(b)
        print('{}, {}'.format(ret_a, ret_b))

    def test_equal(self):
        a = make_dict({'x': 1, 'y': 2}, is_ft=True)
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret = _ffi_api.DictEqual(a, b)
        print('{}'.format(ret))
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2})
        ret = _ffi_api.DictEqual(a, b)
        print('{}'.format(ret))

    def test_iter(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.Dict_Iter(a)
        ret_b = _ffi_api.Dict_Iter(b)

    def test_key_iter(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.Dict_KeyIter(a)
        ret_b = _ffi_api.Dict_KeyIter(b)

    def test_value_iter(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.Dict_ValueIter(a)
        ret_b = _ffi_api.Dict_ValueIter(b)

    def test_item_iter(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.Dict_ItemIter(a)
        ret_b = _ffi_api.Dict_ItemIter(b)

    def test_contains(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.DictContains(a, 'x')
        ret_b = _ffi_api.DictContains(b, 'x')
        print('{}, {}'.format(ret_a, ret_b))

    def test_get_item(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.DictGetItem(a, 'x')
        ret_b = _ffi_api.DictGetItem(b, 'x')
        print('{}, {}'.format(ret_a, ret_b))

    def test_clear(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        _ffi_api.DictClear(a)
        _ffi_api.DictClear(b)
        print('{}, {}'.format(a, b))

    def test_set_item(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        _ffi_api.DictSetItem(a, 'x', 3)
        _ffi_api.DictSetItem(b, 'x', 3)
        print('{}, {}'.format(a, b))

    def test_get_default(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.DictGetDefault(a, 'x')
        ret_b = _ffi_api.DictGetDefault(b, 'x')
        print('{}, {}'.format(ret_a, ret_b))
        ret_a = _ffi_api.DictGetDefault(a, 'z')
        ret_b = _ffi_api.DictGetDefault(b, 'z')
        print('{}, {}'.format(ret_a, ret_b))
        ret_a = _ffi_api.DictGetDefault(a, 'x', 3)
        ret_b = _ffi_api.DictGetDefault(b, 'x', 3)
        print('{}, {}'.format(ret_a, ret_b))
        ret_a = _ffi_api.DictGetDefault(a, 'z', 3)
        ret_b = _ffi_api.DictGetDefault(b, 'z', 3)
        print('{}, {}'.format(ret_a, ret_b))

    def test_pop(self):
        a = make_dict({'x': 1, 'y': 2})
        b = make_dict({'x': 1, 'y': 2}, is_ft=True)
        ret_a = _ffi_api.DictPop(a, 'x')
        ret_b = _ffi_api.DictPop(b, 'x')
        print('{}, {}'.format(ret_a, ret_b))
        with self.assertRaises(Exception):
            _ffi_api.DictPop(a, 'z')
        with self.assertRaises(Exception):
            _ffi_api.DictPop(b, 'z')
        ret_a = _ffi_api.DictPop(a, 'x', 3)
        ret_b = _ffi_api.DictPop(b, 'x', 3)
        print('{}, {}'.format(ret_a, ret_b))
        ret_a = _ffi_api.DictPop(a, 'z', 3)
        ret_b = _ffi_api.DictPop(b, 'z', 3)
        print('{}, {}'.format(ret_a, ret_b))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
