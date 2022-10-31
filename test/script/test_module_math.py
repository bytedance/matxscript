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
import math
from typing import Any


class TestBuiltinMath(unittest.TestCase):

    # TODO: more math test, and accept object as argument

    def test_math_exp(self):
        @matx.script
        def math_exp(a: float) -> float:
            return math.exp(a)

        @matx.script
        def object_math_exp(a: Any) -> float:
            return math.exp(a)

        # @matx.script
        # def math_exp2(a: float) -> float:
        #     return math.exp2(a)
        #
        # @matx.script
        # def object_math_exp2(a: Any) -> float:
        #     return math.exp2(a)

        # @matx.script
        # def math_exp10(a: float) -> float:
        #     return math.exp10(a)
        #
        # @matx.script
        # def object_math_exp10(a: Any) -> float:
        #     return math.exp10(a)

        v = 10.0
        self.assertAlmostEqual(math_exp(v), math.exp(v))
        self.assertAlmostEqual(object_math_exp(v), math.exp(v))
        # self.assertAlmostEqual(math_exp2(v), math.exp2(v))
        # self.assertAlmostEqual(object_math_exp2(v), math.exp2(v))
        # self.assertAlmostEqual(math_exp10(v), math.exp10(v))
        # self.assertAlmostEqual(object_math_exp10(v), math.exp10(v))

    def test_math_pow(self):
        @matx.script
        def math_pow(a: int, b: int) -> Any:
            return math.pow(a, b)

        @matx.script
        def object_math_pow(a: Any, b: int) -> Any:
            return math.pow(a, b)

        self.assertAlmostEqual(math_pow(2, -1), math.pow(2, -1))
        self.assertAlmostEqual(object_math_pow(2, -1), math.pow(2, -1))

        self.assertIsInstance(math_pow(2, -1), float)
        self.assertIsInstance(object_math_pow(2, -1), float)

        with self.assertRaises(Exception) as context:
            math_pow(0, -1)
        with self.assertRaises(Exception) as context:
            object_math_pow(0, -1)

    def test_math_sqrt(self):
        @matx.script
        def math_sqrt(a: int) -> Any:
            return math.sqrt(a)

        @matx.script
        def object_math_sqrt(a: Any) -> Any:
            return math.sqrt(a)

        self.assertAlmostEqual(math_sqrt(2), math.sqrt(2))
        self.assertAlmostEqual(object_math_sqrt(2), math.sqrt(2))

        self.assertIsInstance(math_sqrt(2), float)
        self.assertIsInstance(object_math_sqrt(2), float)

        with self.assertRaises(Exception) as context:
            math_sqrt(-1)
        with self.assertRaises(Exception) as context:
            object_math_sqrt(-1)

    def test_math_log(self):
        @matx.script
        def math_log(a: float) -> float:
            return math.log(a)

        @matx.script
        def object_math_log(a: Any) -> float:
            return math.log(a)

        @matx.script
        def math_log2(a: float) -> float:
            return math.log2(a)

        @matx.script
        def object_math_log2(a: Any) -> float:
            return math.log2(a)

        @matx.script
        def math_log10(a: float) -> float:
            return math.log10(a)

        @matx.script
        def object_math_log10(a: Any) -> float:
            return math.log10(a)

        v = 10.0
        self.assertAlmostEqual(math_log(v), math.log(v))
        self.assertAlmostEqual(object_math_log(v), math.log(v))
        with self.assertRaises(Exception) as context:
            math_log(-10.0)

        self.assertAlmostEqual(math_log2(v), math.log2(v))
        self.assertAlmostEqual(object_math_log2(v), math.log2(v))
        with self.assertRaises(Exception) as context:
            math_log2(-10.0)

        self.assertAlmostEqual(math_log10(v), math.log10(v))
        self.assertAlmostEqual(object_math_log10(v), math.log10(v))
        with self.assertRaises(Exception) as context:
            math_log10(-10.0)

    def test_abs(self):
        @matx.script
        def abs_int(a: int) -> Any:
            return abs(a)

        @matx.script
        def abs_float(a: float) -> Any:
            return abs(a)

        @matx.script
        def object_abs(a: Any) -> Any:
            return abs(a)

        self.assertEqual(abs_int(2), 2)
        self.assertEqual(abs_int(-2), 2)
        self.assertAlmostEqual(abs_float(2.0), 2.0)
        self.assertAlmostEqual(abs_float(-2.0), 2.0)
        self.assertEqual(object_abs(2), 2)
        self.assertEqual(object_abs(-2), 2)
        self.assertAlmostEqual(object_abs(2.0), 2.0)
        self.assertAlmostEqual(object_abs(-2.0), 2.0)

        self.assertIsInstance(abs_int(2), int)
        self.assertIsInstance(abs_float(2.0), float)
        self.assertIsInstance(object_abs(2), int)
        self.assertIsInstance(object_abs(2.0), float)

    def test_math_floor(self):
        @matx.script
        def math_floor(a: float) -> Any:
            return math.floor(a)

        @matx.script
        def object_math_floor(a: Any) -> Any:
            return math.floor(a)

        self.assertEqual(math_floor(2.6), math.floor(2.6))
        self.assertEqual(object_math_floor(2.6), math.floor(2.6))

        self.assertIsInstance(math_floor(2.6), int)
        self.assertIsInstance(object_math_floor(2.6), int)

    def test_math_ceil(self):
        @matx.script
        def math_ceil(a: float) -> Any:
            return math.ceil(a)

        @matx.script
        def object_math_ceil(a: Any) -> Any:
            return math.ceil(a)

        self.assertEqual(math_ceil(2.6), math.ceil(2.6))
        self.assertEqual(object_math_ceil(2.6), math.ceil(2.6))

        self.assertIsInstance(math_ceil(2.6), int)
        self.assertIsInstance(object_math_ceil(2.6), int)

    def test_math_sin(self):
        @matx.script
        def math_sin(a: float) -> Any:
            return math.sin(a)

        @matx.script
        def object_math_sin(a: Any) -> Any:
            return math.sin(a)

        self.assertAlmostEqual(math_sin(math.pi / 6), math.sin(math.pi / 6))
        self.assertAlmostEqual(object_math_sin(math.pi / 6), math.sin(math.pi / 6))

        self.assertIsInstance(math_sin(math.pi / 6), float)
        self.assertIsInstance(object_math_sin(math.pi / 6), float)

    def test_math_cos(self):
        @matx.script
        def math_cos(a: float) -> Any:
            return math.cos(a)

        @matx.script
        def object_math_cos(a: Any) -> Any:
            return math.cos(a)

        self.assertAlmostEqual(math_cos(math.pi / 6), math.cos(math.pi / 6))
        self.assertAlmostEqual(object_math_cos(math.pi / 6), math.cos(math.pi / 6))

        self.assertIsInstance(math_cos(math.pi / 6), float)
        self.assertIsInstance(object_math_cos(math.pi / 6), float)

    def test_math_tan(self):
        @matx.script
        def math_tan(a: float) -> Any:
            return math.tan(a)

        @matx.script
        def object_math_tan(a: Any) -> Any:
            return math.tan(a)

        self.assertAlmostEqual(math_tan(math.pi / 6), math.tan(math.pi / 6))
        self.assertAlmostEqual(object_math_tan(math.pi / 6), math.tan(math.pi / 6))

        self.assertIsInstance(math_tan(math.pi / 6), float)
        self.assertIsInstance(object_math_tan(math.pi / 6), float)

    def test_math_min(self):
        @matx.script
        def sequence_min(a: int, b: float, c: int) -> Any:
            return min(a, b, c)

        @matx.script
        def list_min(a: matx.List) -> Any:
            return min(a)

        @matx.script
        def set_min(a: matx.Set) -> Any:
            return min(a)

        @matx.script
        def object_min(a: Any) -> Any:
            return min(a)

        @matx.script
        def int_min(a: int, b: int, c: int) -> int:
            return min(a, b, c)

        @matx.script
        def double_min(a: float, b: float, c: float) -> float:
            return min(a, b, c)

        self.assertAlmostEqual(sequence_min(2, 1.0, 3), 1.0)
        self.assertEqual(list_min([2, 1, 3]), 1)
        self.assertEqual(set_min({2, 1, 3}), 1)
        self.assertEqual(object_min([2, 1, 3]), 1)
        self.assertEqual(object_min({2, 1, 3}), 1)
        self.assertEqual(int_min(2, 1, 3), 1)
        self.assertAlmostEqual(double_min(2, 1.0, 3), 1.0)

        self.assertIsInstance(sequence_min(2, 1.0, 3), float)
        self.assertIsInstance(object_min([2, 1, 3.0]), int)
        self.assertIsInstance(object_min({2, 1, 3.0}), int)

    def test_math_max(self):
        @matx.script
        def sequence_max(a: int, b: float, c: int) -> Any:
            return max(a, b, c)

        @matx.script
        def list_max(a: matx.List) -> Any:
            return max(a)

        @matx.script
        def set_max(a: matx.Set) -> Any:
            return max(a)

        @matx.script
        def object_max(a: Any) -> Any:
            return max(a)

        @matx.script
        def int_max(a: int, b: int, c: int) -> int:
            return max(a, b, c)

        @matx.script
        def double_max(a: float, b: float, c: float) -> float:
            return max(a, b, c)

        self.assertAlmostEqual(sequence_max(2, 5.0, 3), 5.0)
        self.assertEqual(list_max([2, 5, 3]), 5)
        self.assertEqual(set_max({2, 5, 3}), 5)
        self.assertEqual(object_max([2, 5, 3]), 5)
        self.assertEqual(object_max({2, 5, 3}), 5)
        self.assertAlmostEqual(int_max(2, 5, 3), 5)
        self.assertAlmostEqual(double_max(2, 5.0, 3), 5.0)

        self.assertIsInstance(sequence_max(2, 5.0, 3), float)
        self.assertIsInstance(object_max([2, 5, 3.0]), int)
        self.assertIsInstance(object_max({2, 5, 3.0}), int)

    def test_math_finite(self):
        @matx.script
        def test_inf(x: float) -> bool:
            return math.isinf(x)

        @matx.script
        def test_str_inf(x: str) -> bool:
            return math.isinf(float(x))

        @matx.script
        def test_nan(x: float) -> bool:
            return math.isnan(x)

        @matx.script
        def test_str_nan(x: str) -> bool:
            return math.isnan(float(x))

        @matx.script
        def test_finite(x: float) -> bool:
            return math.isfinite(x)

        @matx.script
        def test_str_finite(x: str) -> bool:
            return math.isfinite(float(x))

        self.assertEqual(test_inf(2.0), False)
        self.assertEqual(test_inf(float('inf')), True)
        self.assertEqual(test_str_inf('inf'), True)

        self.assertEqual(test_nan(2.0), False)
        self.assertEqual(test_nan(float('nan')), True)
        self.assertEqual(test_str_nan('nan'), True)

        self.assertEqual(test_finite(2.0), True)
        self.assertEqual(test_finite(float('inf')), False)
        self.assertEqual(test_str_finite('inf'), False)
        self.assertEqual(test_finite(float('nan')), False)
        self.assertEqual(test_str_finite('nan'), False)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
