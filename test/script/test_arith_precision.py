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
from typing import Any, Tuple, List


class TestArithPrecision(unittest.TestCase):

    def test_int_floormod(self):
        def my_func_floormod(a: int, b: int) -> int:
            return a % b

        x = 9223372036854775807
        y = 9223372036854775807 // 2 + 1

        py_ret = my_func_floormod(x, y)
        tx_ret = matx.script(my_func_floormod)(x, y)
        self.assertEqual(py_ret, tx_ret)

    def test_int_floordiv(self):
        def my_func_floordiv(a: int, b: int) -> int:
            return a // b

        x = 9223372036854775807
        y = 2

        py_ret = my_func_floordiv(x, y)
        tx_ret = matx.script(my_func_floordiv)(x, y)
        self.assertEqual(py_ret, tx_ret)

    def test_float_floordiv(self):
        def my_func_floordiv(a: float, b: float) -> float:
            return a // b

        tx_func = matx.script(my_func_floordiv)

        data = ((5.1, 3.2), (-5.1, 3.2), (5.1, -3.2), (-5.1, -3.2),)
        for td in data:
            py_ret = my_func_floordiv(td[0], td[1])
            tx_ret = tx_func(td[0], td[1])
            self.assertEqual(py_ret, tx_ret)

    def test_float_floormod(self):
        def my_func_floormod(a: float, b: float) -> float:
            return a % b

        tx_func = matx.script(my_func_floormod)

        data = ((5.0, 3.0), (5.0, -3.0), (-5.0, 3.0), (-5.0, -3.0),)
        for td in data:
            py_ret = my_func_floormod(td[0], td[1])
            tx_ret = tx_func(td[0], td[1])
            self.assertEqual(py_ret, tx_ret)

    def test_mixed_floormod(self):
        def my_func_floormod_i64_d64(a: int, b: float) -> Any:
            return a % b

        def my_func_floormod_d64_i64(a: float, b: int) -> Any:
            return a % b

        def my_func_floormod_i64_any(a: int, b: Any) -> Any:
            return a % b

        def my_func_floormod_any_i64(a: Any, b: int) -> Any:
            return a % b

        def my_func_floormod_d64_any(a: float, b: Any) -> Any:
            return a % b

        def my_func_floormod_any_d64(a: Any, b: float) -> Any:
            return a % b

        test_funcs = [
            my_func_floormod_i64_d64,
            my_func_floormod_d64_i64,
            my_func_floormod_i64_any,
            my_func_floormod_any_i64,
            my_func_floormod_d64_any,
            my_func_floormod_any_d64,
        ]

        for my_func in test_funcs:
            tx_func = matx.script(my_func)

            data = ((5, 3), (5, -3), (-5, 3), (-5, -3),)
            for td in data:
                py_ret = my_func(td[0], td[1])
                tx_ret = tx_func(td[0], td[1])
                self.assertEqual(py_ret, tx_ret)

    def test_mixed_floordiv(self):
        def my_func_floordiv_i64_d64(a: int, b: float) -> Any:
            return a // b

        def my_func_floordiv_d64_i64(a: float, b: int) -> Any:
            return a // b

        def my_func_floordiv_i64_any(a: int, b: Any) -> Any:
            return a // b

        def my_func_floordiv_any_i64(a: Any, b: int) -> Any:
            return a // b

        def my_func_floordiv_d64_any(a: float, b: Any) -> Any:
            return a // b

        def my_func_floordiv_any_d64(a: Any, b: float) -> Any:
            return a // b

        test_funcs = [
            my_func_floordiv_i64_d64,
            my_func_floordiv_d64_i64,
            my_func_floordiv_i64_any,
            my_func_floordiv_any_i64,
            my_func_floordiv_d64_any,
            my_func_floordiv_any_d64,
        ]

        for my_func in test_funcs:
            tx_func = matx.script(my_func)

            data = ((5, 3), (5, -3), (-5, 3), (-5, -3),)
            for td in data:
                py_ret = my_func(td[0], td[1])
                tx_ret = tx_func(td[0], td[1])
                self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
