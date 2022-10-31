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

import os
import unittest
import matx

from typing import Any


class TestIsinstance(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return super().setUp()

    def test_is_user_cls(self):
        class MyObj:
            def __init__(self):
                pass

        def is_user_cls(a: Any) -> bool:
            return isinstance(a, MyObj)

        with self.assertRaises(Exception) as ctx:
            matx.script(is_user_cls)

    def test_is_bool(self):
        def is_bool(a: Any) -> bool:
            return isinstance(a, bool)

        with self.assertRaises(Exception) as ctx:
            matx.script(is_bool)

    def test_is_int(self):
        def is_int(a: Any) -> bool:
            return isinstance(a, int)

        c = (1, "s")
        py_ret = tuple([is_int(i) for i in c])
        tx_is_int = matx.script(is_int)
        tx_ret = tuple([tx_is_int(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_is_float(self):
        def is_float(a: Any) -> bool:
            return isinstance(a, float)

        c = (1.1, "s")
        py_ret = tuple([is_float(i) for i in c])
        tx_is_float = matx.script(is_float)
        tx_ret = tuple([tx_is_float(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_explicit_is_prim(self):
        def explicit_is_prim() -> Any:
            return (isinstance(1, int),
                    isinstance(1, float),
                    isinstance(1.1, int),
                    isinstance(1.1, float))

        py_ret = explicit_is_prim()
        tx_ret = matx.script(explicit_is_prim)()
        self.assertEqual(py_ret, tx_ret)

    def test_is_str(self):
        def is_str(a: Any) -> bool:
            return isinstance(a, str)

        c = (1.1, "s")
        py_ret = tuple([is_str(i) for i in c])
        tx_is_str = matx.script(is_str)
        tx_ret = tuple([tx_is_str(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_is_bytes(self):
        def is_bytes(a: Any) -> bool:
            return isinstance(a, bytes)

        c = (1.1, "s", b"bytes")
        py_ret = tuple([is_bytes(i) for i in c])
        tx_is_bytes = matx.script(is_bytes)
        tx_ret = tuple([tx_is_bytes(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_is_list(self):
        def is_list(a: Any) -> bool:
            return isinstance(a, list)

        c = (1.1, ["s", b"bytes"])
        py_ret = tuple([is_list(i) for i in c])
        tx_is_list = matx.script(is_list)
        tx_ret = tuple([tx_is_list(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_is_tuple(self):
        def is_tuple(a: Any) -> bool:
            return isinstance(a, tuple)

        c = (1.1, ("s", b"bytes"))
        py_ret = tuple([is_tuple(i) for i in c])
        tx_is_tuple = matx.script(is_tuple)
        tx_ret = tuple([tx_is_tuple(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_is_set(self):
        def is_set(a: Any) -> bool:
            return isinstance(a, set)

        c = (1.1, {"s", b"bytes"})
        py_ret = tuple([is_set(i) for i in c])
        tx_is_set = matx.script(is_set)
        tx_ret = tuple([tx_is_set(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_is_dict(self):
        def is_dict(a: Any) -> bool:
            return isinstance(a, dict)

        c = (1.1, {"s": b"bytes"})
        py_ret = tuple([is_dict(i) for i in c])
        tx_is_dict = matx.script(is_dict)
        tx_ret = tuple([tx_is_dict(i) for i in c])
        self.assertEqual(py_ret, tx_ret)

    def test_is_multi_type(self):
        def is_multi_type(a: Any) -> bool:
            return isinstance(a, (float, int, set, dict))

        c = (1, 1.1, {"s"}, {"s": b"bytes"})
        py_ret = tuple([is_multi_type(i) for i in c])
        tx_is_multi_type = matx.script(is_multi_type)
        tx_ret = tuple([tx_is_multi_type(i) for i in c])
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
