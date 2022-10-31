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

from typing import Tuple, List, Any
import os
import unittest
import matx


class TestBuiltinUnpack(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return super().setUp()

    def test_runtime_unpack_list(self):
        al = [1, 2, 3]
        a, b, c = al
        print(a, b, c)

    def test_runtime_unpack_tuple(self):
        al = (1, 2, 3)
        a, b, c = al
        print(a, b, c)

    def test_codegen_unpack_list(self):
        @matx.script
        def test_unpack_list() -> Tuple[int, int, int]:
            al = [1, 2, 3]
            a, b, c = al
            return a, b, c

        ret1, ret2, ret3 = test_unpack_list()
        self.assertEqual(ret1, 1)
        self.assertEqual(ret2, 2)
        self.assertEqual(ret3, 3)

    def test_codegen_unpack_tuple(self):
        @matx.script
        def test_unpack_tuple() -> Tuple[int, int, int]:
            al = (1, 2, 3)
            d, (e, f), g = (1, (2, 3), 4)
            a, b, c = al
            return a, b, c

        result = test_unpack_tuple()
        self.assertEqual(len(result), 3)

    def test_codegen_unroll_tuple(self):
        @matx.script
        def test_unroll_tuple() -> int:
            a, (b, c), d = 1, (2, 3), 4
            return a + b + c + d

        self.assertEqual(test_unroll_tuple(), 10)

        @matx.script
        def test_swap_tuple() -> int:
            a = 1
            b = 2
            a, b = b, a
            return a * 10 + b

        self.assertEqual(test_swap_tuple(), 21)

    def test_unpack_in_for_ctx(self):
        def test_unpack_list_in_for_ctx(int_list: List[List[List[int]]]) -> List[int]:
            ret_list: List[int] = []
            for a, b in int_list:
                ret_list.append(a[0])
                ret_list.append(b[0])
            return ret_list

        my_test_data = [[[1], [2]]]
        py_ret = test_unpack_list_in_for_ctx(my_test_data)
        tx_ret = matx.script(test_unpack_list_in_for_ctx)(my_test_data)
        self.assertEqual(py_ret, tx_ret)

        def test_unpack_tuple_in_for_ctx(int_list: List[Tuple[List[int], List[int]]]) -> List[int]:
            ret_list: List[int] = []
            for a, b in int_list:
                ret_list.append(a[0])
                ret_list.append(b[0])
            return ret_list

        my_test_data = [([1], [2])]
        py_ret = test_unpack_tuple_in_for_ctx(my_test_data)
        tx_ret = matx.script(test_unpack_tuple_in_for_ctx)(my_test_data)
        self.assertEqual(py_ret, tx_ret)

        def test_unpack_any_in_for_ctx(int_list: List[Any]) -> List[int]:
            ret_list: List[int] = []
            for a, b in int_list:
                ret_list.append(a[0])
                ret_list.append(b[0])
            return ret_list

        my_test_data = [([1], [2])]
        py_ret = test_unpack_any_in_for_ctx(my_test_data)
        tx_ret = matx.script(test_unpack_any_in_for_ctx)(my_test_data)
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
