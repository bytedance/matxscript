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
from matx import FTList
from typing import Any, Tuple, List


class TestMatxArith(unittest.TestCase):

    def test_basic_arith(self):

        @matx.script
        def a_plus_b(a: int, b: int) -> int:
            c = matx.List([123])
            y: int = c[0]
            return a + b

        @matx.script
        def object_sequence_plus_float(a: Any) -> Any:
            return a + 2.0 + 1.0

        @matx.script
        def object_sequence_plus_int(a: Any) -> Any:
            return a + 2 + 1

        @matx.script
        def a_mul_b(a: int, b: int) -> int:
            return a * b

        @matx.script
        def object_sequence_mul_float(a: Any) -> Any:
            return a * 2.0 * 1.0

        @matx.script
        def object_sequence_mul_int(a: Any) -> Any:
            return a * 2 * 1

        @matx.script
        def a_minus_b(a: int, b: int) -> int:
            return a - b

        @matx.script
        def object_sequence_minus_float(a: Any) -> Any:
            return a - 2.0 - 1.0

        @matx.script
        def object_sequence_minus_int(a: Any) -> Any:
            return a - 2 - 1

        @matx.script
        def int_a_div_int_b(a: int, b: int) -> Any:
            return a / b

        @matx.script
        def int_a_div_float_b(a: int, b: float) -> Any:
            return a / b

        @matx.script
        def object_a_div_b(a: Any, b: int) -> Any:
            return a / b

        @matx.script
        def object_sequence_div_float(a: Any) -> Any:
            return a / 2.0 / 1.0

        @matx.script
        def object_sequence_div_int(a: Any) -> Any:
            return a / 2 / 1

        @matx.script
        def int_a_floordiv_int_b(a: int, b: int) -> Any:
            return a // b

        @matx.script
        def int_a_floordiv_float_b(a: int, b: float) -> Any:
            return a // b

        @matx.script
        def object_a_floordiv_b(a: Any, b: int) -> Any:
            return a // b

        @matx.script
        def object_sequence_floordiv_float(a: Any) -> Any:
            return a // 2.0 // 1.0

        @matx.script
        def object_sequence_floordiv_int(a: Any) -> Any:
            return a // 2 // 1

        @matx.script
        def int_a_floormod_int_b(a: int, b: int) -> Any:
            return a % b

        @matx.script
        def int_a_floormod_float_b(a: int, b: float) -> Any:
            return a % b

        @matx.script
        def object_a_floormod_b(a: Any, b: int) -> Any:
            return a % b

        @matx.script
        def object_sequence_floormod_float(a: Any) -> Any:
            return a % 2.0 % 1.0

        @matx.script
        def object_sequence_floormod_int(a: Any) -> Any:
            return a % 2 % 1

        import math

        self.assertEqual(a_plus_b(2, 2), 4)
        self.assertEqual(a_mul_b(2, 3), 6)
        self.assertEqual(a_minus_b(1, 2), -1)

        self.assertAlmostEqual(object_sequence_plus_float(5.0), 8.0)
        self.assertAlmostEqual(object_sequence_mul_float(5.0), 10.0)
        self.assertAlmostEqual(object_sequence_minus_float(5.0), 2.0)
        self.assertEqual(object_sequence_plus_int(5), 8)
        self.assertEqual(object_sequence_mul_int(5), 10)
        self.assertEqual(object_sequence_minus_int(5), 2)

        self.assertAlmostEqual(int_a_div_int_b(5, 2), 2.5)
        self.assertAlmostEqual(int_a_div_float_b(5, 2.0), 2.5)
        self.assertAlmostEqual(object_a_div_b(5, 2), 2.5)
        self.assertAlmostEqual(object_a_div_b(5.0, 2), 2.5)
        self.assertAlmostEqual(object_sequence_div_float(5.0), 2.5)
        self.assertAlmostEqual(object_sequence_div_int(5), 2.5)

        self.assertEqual(int_a_floordiv_int_b(5, 2), 2)
        self.assertAlmostEqual(int_a_floordiv_float_b(5, 2.0), 2.0)
        self.assertEqual(object_a_floordiv_b(5, 2), 2)
        self.assertAlmostEqual(object_a_floordiv_b(5.0, 2), 2.0)
        self.assertAlmostEqual(object_sequence_floordiv_float(5.0), 2.0)
        self.assertEqual(object_sequence_floordiv_int(5), 2)

        self.assertEqual(int_a_floormod_int_b(5, 2), 1)
        self.assertAlmostEqual(int_a_floormod_float_b(5, 2.0), 1.0)
        self.assertEqual(object_a_floormod_b(5, 2), 1)
        self.assertAlmostEqual(object_a_floormod_b(5.0, 2), 1.0)
        self.assertAlmostEqual(object_sequence_floormod_float(5.0), 0.0)
        self.assertEqual(object_sequence_floormod_int(5), 0)

        self.assertIsInstance(int_a_div_int_b(5, 2), float)
        self.assertIsInstance(int_a_div_float_b(5, 2.0), float)
        self.assertIsInstance(object_a_div_b(5, 2), float)
        self.assertIsInstance(object_a_div_b(5.0, 2), float)

        self.assertIsInstance(int_a_floordiv_int_b(5, 2), int)
        self.assertIsInstance(int_a_floordiv_float_b(5, 2.0), float)
        self.assertIsInstance(object_a_floordiv_b(5, 2), int)
        self.assertIsInstance(object_a_floordiv_b(5.0, 2), float)

        self.assertIsInstance(int_a_floormod_int_b(5, 2), int)
        self.assertIsInstance(int_a_floormod_float_b(5, 2.0), float)
        self.assertIsInstance(object_a_floormod_b(5, 2), int)
        self.assertIsInstance(object_a_floormod_b(5.0, 2), float)

        with self.assertRaises(Exception) as context:
            int_a_div_float_b(5, 0.0)
        with self.assertRaises(Exception) as context:
            object_a_div_b(5, 0)
        with self.assertRaises(Exception) as context:
            int_a_floordiv_float_b(5, 0.0)
        with self.assertRaises(Exception) as context:
            object_a_floordiv_b(5.0, 0)
        with self.assertRaises(Exception) as context:
            int_a_floormod_float_b(5, 0.0)
        with self.assertRaises(Exception) as context:
            object_a_floormod_b(5.0, 0)

    def test_nested_arith(self):

        @matx.script
        def abc_calc(a: int, b: int, c: int) -> int:
            return 4 * a + b // 2 - (a - b) + (b + c) + b // 2

        @matx.script
        def assign_calc(a: int, b: int) -> int:
            c: int = 4 * b
            return a + c

        self.assertEqual(abc_calc(1, 2, 3), 12)
        self.assertEqual(assign_calc(2, 3), 14)

    def test_chaining_comparision(self):
        @matx.script
        def chaining_lt(a: int, b: int, c: int) -> bool:
            return a < b < c

        self.assertTrue(chaining_lt(1, 2, 3))

    def test_bit_operation(self):
        @matx.script
        def bitwise_and(a: Any, b: Any) -> Any:
            return a & b

        @matx.script
        def bitwise_or(a: Any, b: Any) -> Any:
            return a | b

        @matx.script
        def bitwise_xor(a: Any, b: Any) -> Any:
            return a ^ b

        @matx.script
        def bitwise_not(a: Any) -> Any:
            return ~a

        @matx.script
        def left_shift(a: Any, b: Any) -> Any:
            return a << b

        @matx.script
        def right_shift(a: Any, b: Any) -> Any:
            return a >> b

        c = matx.List([1, 2])

        self.assertEqual(bitwise_and(2, 1), 0)
        self.assertEqual(bitwise_and(2, c[0]), 0)
        self.assertEqual(bitwise_or(2, 1), 3)
        self.assertEqual(bitwise_or(2, c[0]), 3)
        self.assertEqual(bitwise_xor(2, 1), 3)
        self.assertEqual(bitwise_xor(2, c[0]), 3)
        self.assertEqual(bitwise_not(2), -3)
        self.assertEqual(bitwise_not(c[1]), -3)

        self.assertEqual(left_shift(2, 1), 4)
        self.assertEqual(left_shift(2, c[0]), 4)
        self.assertEqual(right_shift(2, 1), 1)
        self.assertEqual(right_shift(2, c[0]), 1)

        self.assertIsInstance(bitwise_and(2, 1), int)
        self.assertIsInstance(bitwise_and(2, c[0]), int)
        self.assertIsInstance(bitwise_or(2, 1), int)
        self.assertIsInstance(bitwise_or(2, c[0]), int)
        self.assertIsInstance(bitwise_xor(2, 1), int)
        self.assertIsInstance(bitwise_xor(2, c[0]), int)
        self.assertIsInstance(bitwise_not(2), int)
        self.assertIsInstance(bitwise_not(c[1]), int)

        self.assertIsInstance(left_shift(2, 1), int)
        self.assertIsInstance(left_shift(2, c[0]), int)
        self.assertIsInstance(right_shift(2, 1), int)
        self.assertIsInstance(right_shift(2, c[0]), int)

    def test_advance_logic(self):
        def test_int_and() -> int:
            return 3 and 5

        def test_str_and1() -> str:
            return "hello" and "world"

        def test_str_and2() -> str:
            a = "hello"
            return a and "world"

        def test_bytes_and1() -> bytes:
            return b"hello" and b"world"

        def test_bytes_and2() -> bytes:
            a = b"hello"
            return a and b"world"

        def test_any_and() -> Any:
            return [2] and 5

        self.assertEqual(test_int_and(), matx.script(test_int_and)())
        self.assertEqual(test_str_and1(), matx.script(test_str_and1)())
        self.assertEqual(test_str_and2(), matx.script(test_str_and2)())
        self.assertEqual(test_bytes_and1(), matx.script(test_bytes_and1)())
        self.assertEqual(test_bytes_and2(), matx.script(test_bytes_and2)())
        self.assertEqual(test_any_and(), matx.script(test_any_and)())

        def test_int_or() -> int:
            return 3 or 5

        def test_str_or1() -> str:
            return "hello" or "world"

        def test_str_or2() -> str:
            a = "hello"
            return a or "world"

        def test_bytes_or1() -> bytes:
            return b"hello" or b"world"

        def test_bytes_or2() -> bytes:
            a = b"hello"
            return a or b"world"

        def test_any_or() -> Any:
            return '' or {4: 5}

        self.assertEqual(test_int_or(), matx.script(test_int_or)())
        self.assertEqual(test_str_or1(), matx.script(test_str_or1)())
        self.assertEqual(test_str_or2(), matx.script(test_str_or2)())
        self.assertEqual(test_bytes_or1(), matx.script(test_bytes_or1)())
        self.assertEqual(test_bytes_or2(), matx.script(test_bytes_or2)())
        self.assertEqual(test_any_or(), matx.script(test_any_or)())

        def test_and_or() -> Any:
            a = 1 and '' or []
            b = 0 and '1' or [2]
            return a, b

        self.assertEqual(test_and_or(), matx.script(test_and_or)())

    def test_container_comp(self):
        def generic_comp(a: Any, b: Any) -> Any:
            return a > b, a >= b, a < b, a <= b

        def list_comp(a: List, b: List) -> Any:
            return a > b, a >= b, a < b, a <= b

        def ftlist_comp(aa: List[int], bb: List[int]) -> Any:
            a: FTList[int] = []
            for x in aa:
                a.append(x)
            b: FTList[int] = []
            for x in bb:
                b.append(x)
            return a > b, a >= b, a < b, a <= b

        def tuple_comp(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Any:
            return a > b, a >= b, a < b, a <= b

        generic_comp_op = matx.script(generic_comp)
        list_comp_op = matx.script(list_comp)
        ftlist_comp_op = matx.script(ftlist_comp)
        tuple_comp_op = matx.script(tuple_comp)

        self.assertEqual(generic_comp((1, 2, 3), (1, 2, 4)),
                         generic_comp_op((1, 2, 3), (1, 2, 4)))
        self.assertEqual(tuple_comp((1, 2, 3), (1, 2, 4)),
                         tuple_comp_op((1, 2, 3), (1, 2, 4)))

        self.assertEqual(generic_comp([1, 2, 3], [1, 2, 4]),
                         generic_comp_op([1, 2, 3], [1, 2, 4]))
        self.assertEqual(list_comp([1, 2, 3], [1, 2, 4]),
                         list_comp_op([1, 2, 3], [1, 2, 4]))
        self.assertEqual(ftlist_comp([1, 2, 3], [1, 2, 4]),
                         ftlist_comp_op([1, 2, 3], [1, 2, 4]))

        # TODO: fix me
        # def wrapper() -> Any:
        #     a = (1, 2, 3, 4) < (1, 2, 4)
        #     b = (1, 2) < (1, 2, -1)
        #     c = (1, 2, 3) == (1.0, 2.0, 3.0)
        #     d = (1, 2, ('aa', 'ab')) < (1, 2, ('abc', 'a'), 4)
        #     return a, b, c, d
        # self.assertEqual(wrapper(), matx.script(wrapper)())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
