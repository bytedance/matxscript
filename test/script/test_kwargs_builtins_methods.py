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
import matx
from typing import Any


class TestBuiltinsMethodsKeywordArgs(unittest.TestCase):

    def test_list(self):
        # Only the sort method supports kwargs, so we just need to test the other methods
        def test_list_append() -> None:
            a = list()
            a.append(__object=1)

        with self.assertRaises(Exception):
            matx.script(test_list_append)

    def test_dict(self):
        # Only the update method supports kwargs
        def test_dict_pop() -> None:
            a = {1: 2}
            a.pop(key=1)

        with self.assertRaises(Exception):
            matx.script(test_dict_pop)

        # def test_dict_update() -> Any:
        #     a = {}
        #     a.update(k=0)
        #     a.update(i=1, j=2)
        #     return a
        #
        # py_ret = test_dict_update()
        # tx_ret = matx.script(test_dict_update)()
        # self.assertEqual(py_ret, tx_ret)

    def test_set(self):
        # All methods don't supports kwargs
        def test_set_pop() -> None:
            a = {1, 2, 3}
            a.remove(element=1)

        with self.assertRaises(Exception):
            matx.script(test_set_pop)

    def test_tuple(self):
        # All methods don't supports kwargs
        def test_tuple_index() -> None:
            a = (1, 2, 3)
            a.index(__value=1)

        with self.assertRaises(Exception):
            matx.script(test_tuple_index)

    def test_str(self):
        # Support kwargs: expandtabs rsplit split splitlines format encode
        def test_str_index() -> None:
            a = "hello world"
            a.index(sub="world")

        with self.assertRaises(Exception):
            matx.script(test_str_index)

        def test_str_split() -> Any:
            a = "hello world"
            return a.split(maxsplit=0)

        py_ret = test_str_split()
        tx_ret = matx.script(test_str_split)()
        self.assertEqual(py_ret, tx_ret)

        def test_any_split() -> Any:
            a: Any = "hello world"
            return a.split(maxsplit=0)

        py_ret = test_any_split()
        tx_ret = matx.script(test_any_split)()
        self.assertEqual(py_ret, tx_ret)

        # TODO: fix encode
        def test_str_encode() -> Any:
            a = "hello world"
            # return a.encode(encoding="utf-8")
            return a.encode()

        py_ret = test_str_encode()
        tx_ret = matx.script(test_str_encode)()
        self.assertEqual(py_ret, tx_ret)

        # # TODO: fix format
        # def test_str_format() -> Any:
        #     a = "{k} {v}"
        #     return a.format(k=1, v=2)
        #
        # py_ret = test_str_format()
        #
        # tx_ret = matx.script(test_str_format)()
        # self.assertEqual(py_ret, tx_ret)

        # TODO: support splitlines
        def test_str_splitlines() -> None:
            a = "hello world"
            a.splitlines(keepends=True)

        # TODO: support rsplit
        def test_str_rsplit() -> None:
            a = "hello world"
            a.rsplit(maxsplit=0)

        # TODO: support expandtabs
        def test_str_expandtabs() -> None:
            a = "hello world"
            a.expandtabs(tabsize=8)

    def test_bytes(self):
        # Support kwargs: hex split rsplit translate expandtabs splitlines
        def test_bytes_index() -> None:
            a = b"hello world"
            a.index(sub=b"world")

        with self.assertRaises(Exception):
            matx.script(test_bytes_index)

        def test_bytes_split() -> Any:
            a = b"hello world"
            return a.split(maxsplit=0)

        py_ret = test_bytes_split()
        tx_ret = matx.script(test_bytes_split)()
        self.assertEqual(py_ret, tx_ret)

        def test_any_split() -> Any:
            a: Any = b"hello world"
            return a.split(maxsplit=0)

        py_ret = test_any_split()
        tx_ret = matx.script(test_any_split)()
        self.assertEqual(py_ret, tx_ret)

        # TODO: fix encode
        def test_bytes_encode() -> Any:
            a = b"hello world"
            # return a.decode(encoding="utf-8")
            return a.decode()

        py_ret = test_bytes_encode()
        tx_ret = matx.script(test_bytes_encode)()
        self.assertEqual(py_ret, tx_ret)

        # TODO: support splitlines
        def test_bytes_splitlines() -> None:
            a = b"hello world"
            a.splitlines(keepends=True)

        # TODO: support rsplit
        def test_bytes_rsplit() -> None:
            a = b"hello world"
            a.rsplit(maxsplit=0)

        # TODO: support expandtabs
        def test_bytes_expandtabs() -> None:
            a = b"hello world"
            a.expandtabs(tabsize=8)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
