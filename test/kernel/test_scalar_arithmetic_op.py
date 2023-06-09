#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import unittest

from matx.kernel.kernel_parser import KernelParser
from matx.kernel.compile_linalg import compile_linalg
from matx.kernel.typing import int32, float32, float64, boolean


class TestMLIRIntArithmeticOp(unittest.TestCase):
    def helper(self, foo):
        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        f = compile_linalg(p)
        return f

    def test_int_add(self):
        def foo(a: int32, b: int32) -> int32:
            return a + b

        foo = self.helper(foo)

        self.assertEqual(foo(0, 1), 1)
        self.assertEqual(foo(-1, 1), 0)
        self.assertEqual(foo(-5, -8), -13)
        self.assertEqual(foo(5, 7), 12)
        # overflow case
        self.assertEqual(foo(2147483647, 1), -2147483648)
        self.assertEqual(foo(-2147483648, -1), 2147483647)

    def test_int_sub(self):
        def foo(a: int32, b: int32) -> int32:
            return a - b

        foo = self.helper(foo)

        self.assertEqual(foo(0, 1), -1)
        self.assertEqual(foo(-1, 1), -2)
        self.assertEqual(foo(1, 1), 0)
        self.assertEqual(foo(-5, -8), 3)
        self.assertEqual(foo(5, 7), -2)

    def test_int_mul(self):
        def foo(a: int32, b: int32) -> int32:
            return a * b

        foo = self.helper(foo)
        self.assertEqual(foo(0, 1), 0)
        self.assertEqual(foo(-1, 1), -1)
        self.assertEqual(foo(1, 1), 1)
        self.assertEqual(foo(-5, -8), 40)
        self.assertEqual(foo(5, 7), 35)

    def test_int_div(self):
        # numpy int32/int32 = float 64
        def foo(a: int32, b: int32) -> float64:
            return a / b

        foo = self.helper(foo)
        self.assertEqual(foo(0, 1), 0 / 1)
        self.assertEqual(foo(-1, 1), -1 / 1)
        self.assertEqual(foo(1, 1), 1 / 1)
        self.assertEqual(foo(-5, -8), -5 / -8)
        self.assertEqual(foo(5, 7), 5 / 7)

    def test_int_rem(self):
        def foo(a: int32, b: int32) -> int32:
            return a % b

        foo = self.helper(foo)
        self.assertEqual(foo(-10, 3), -10 % 3)
        self.assertEqual(foo(0, 1), 0 % 1)
        self.assertEqual(foo(-1, 1), -1 % 1)
        self.assertEqual(foo(1, 1), 1 % 1)
        self.assertEqual(foo(-5, -8), -5 % -8)
        self.assertEqual(foo(5, -8), 5 % -8)
        self.assertEqual(foo(5, 7), 5 % 7)
        self.assertEqual(foo(-5, 7), -5 % 7)
        self.assertEqual(foo(18328, 32202), 18328 % 32202)
        self.assertEqual(foo(32202, 18328), 32202 % 18328)
        self.assertEqual(foo(-18328, 32202), -18328 % 32202)
        self.assertEqual(foo(18328, -32202), 18328 % -32202)

    def test_int_floordiv(self):
        def foo(a: int32, b: int32) -> int32:
            return a // b

        foo = self.helper(foo)
        self.assertEqual(foo(0, 1), 0 // 1)
        self.assertEqual(foo(-1, 1), -1 // 1)
        self.assertEqual(foo(1, 1), 1 // 1)
        self.assertEqual(foo(-5, -8), -5 // -8)
        self.assertEqual(foo(5, 7), 5 // 7)
        self.assertEqual(foo(5, -2), -3)
        self.assertEqual(foo(5, 2), 2)

    def test_int_min(self):
        def foo(a: int32, b: int32) -> int32:
            return min(a, b)
        # todo not supported yet
        # foo = self.helper(foo)

    def test_int_max(self):
        def foo(a: int32, b: int32) -> int32:
            return max(a, b)
        # todo not supported yet
        # foo = self.helper(foo)


class TestMLIRFloatArithmeticOp(unittest.TestCase):
    def helper(self, foo):
        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        f = compile_linalg(p)
        return f

    def test_float_add(self):
        def foo(a: float64, b: float64) -> float64:
            return a + b

        foo = self.helper(foo)

        self.assertAlmostEquals(foo(0, 1.213), 1.213)
        self.assertAlmostEquals(foo(0, 1.213), 1.213)
        self.assertAlmostEquals(foo(0.1, 1.213), 1.313)
        self.assertAlmostEquals(foo(-1.1, 1.1), 0)
        self.assertAlmostEquals(foo(-5.9, -8.0001), -13.9001)
        self.assertAlmostEquals(foo(5.34, 7.287953), 5.34 + 7.287953)

    def test_float_sub(self):
        def foo(a: float64, b: float64) -> float64:
            return a - b

        foo = self.helper(foo)

        self.assertAlmostEquals(foo(0, 1.213), -1.213)
        self.assertAlmostEquals(foo(0, 1.213), -1.213)
        self.assertAlmostEquals(foo(0.1, 1.213), 0.1 - 1.213)
        self.assertAlmostEquals(foo(-1.1, 1.1), -1.1 - 1.1)
        self.assertAlmostEquals(foo(-5.9, -8.0001), -5.9 - (-8.0001))
        self.assertAlmostEquals(foo(5.34, 7.287953), 5.34 - 7.287953)

    def test_float_mul(self):
        def foo(a: float64, b: float64) -> float64:
            return a * b

        foo = self.helper(foo)
        self.assertAlmostEquals(foo(0, 1.213), 0 * 1.213)
        self.assertAlmostEquals(foo(0, 1.213), 0 * 1.213)
        self.assertAlmostEquals(foo(0.1, 1.213), 0.1 * 1.213)
        self.assertAlmostEquals(foo(-1.1, 1.1), -1.1 * 1.1)
        self.assertAlmostEquals(foo(-5.9, -8.0001), -5.9 * (-8.0001))
        self.assertAlmostEquals(foo(5.34, 7.287953), 5.34 * 7.287953)

    def test_float_div(self):
        # numpy int32/int32 = float 64
        def foo(a: float64, b: float64) -> float64:
            return a / b

        foo = self.helper(foo)
        self.assertAlmostEquals(foo(0, 1.213), 0 / 1.213)
        self.assertAlmostEquals(foo(0, 1.213), 0 / 1.213)
        self.assertAlmostEquals(foo(0.1, 1.213), 0.1 / 1.213)
        self.assertAlmostEquals(foo(-1.1, 1.1), -1.1 / 1.1)
        self.assertAlmostEquals(foo(-5.9, -8.0001), -5.9 / (-8.0001))
        self.assertAlmostEquals(foo(5.34, 7.287953), 5.34 / 7.287953)

    def test_float_rem(self):
        def foo(a: float64, b: float64) -> float64:
            return a % b

        foo = self.helper(foo)
        self.assertEqual(foo(10, -3), 10 % -3)
        self.assertEqual(foo(-10, 3), -10 % 3)
        self.assertEqual(foo(0, 1.1), 0 % 1.1)
        self.assertEqual(foo(-1.1, 1.2), -1.1 % 1.2)
        self.assertEqual(foo(1, 1), 1 % 1)
        self.assertEqual(foo(-5, -8), -5 % -8)
        self.assertEqual(foo(5, -8), 5 % -8)
        self.assertEqual(foo(5.5, -8.3), 5.5 % -8.3)
        self.assertEqual(foo(5, 7), 5 % 7)
        self.assertEqual(foo(-5, 7), -5 % 7)
        self.assertEqual(foo(-5.34, 7.68), -5.34 % 7.68)
        self.assertEqual(foo(18328, 32202), 18328 % 32202)
        self.assertEqual(foo(32202, 18328), 32202 % 18328)
        self.assertEqual(foo(-18328, 32202), -18328 % 32202)
        self.assertEqual(foo(-18328.3534, 32202.3534), -18328.3534 % 32202.3534)
        self.assertEqual(foo(18328, -32202), 18328 % -32202)

    def test_float_floordiv(self):
        # numpy float64/float64 = float 64
        def foo(a: float64, b: float64) -> float64:
            return a // b

        foo = self.helper(foo)
        self.assertAlmostEquals(foo(0, 1.213), 0 // 1.213)
        self.assertAlmostEquals(foo(0, 1.213), 0 // 1.213)
        self.assertAlmostEquals(foo(0.1, 1.213), 0.1 // 1.213)
        self.assertAlmostEquals(foo(-1.1, 1.1), -1.1 // 1.1)
        self.assertAlmostEquals(foo(-5.9, -8.0001), -5.9 // (-8.0001))
        self.assertAlmostEquals(foo(5.34, 7.287953), 5.34 // 7.287953)
        self.assertEqual(foo(5, -2), -3)
        self.assertEqual(foo(5, 2), 2)

    def test_float_min(self):
        def foo(a: float64, b: float64) -> float64:
            return min(a, b)
        # todo not supported yet
        # foo = self.helper(foo)

    def test_float_max(self):
        def foo(a: float64, b: float64) -> float64:
            return max(a, b)
        # todo not supported yet
        # foo = self.helper(foo)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
