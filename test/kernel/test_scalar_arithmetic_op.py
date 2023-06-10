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

        self.assertEqual(1, foo(0, 1))
        self.assertEqual(0, foo(-1, 1))
        self.assertEqual(-13, foo(-5, -8))
        self.assertEqual(12, foo(5, 7))
        # overflow case
        self.assertEqual(-2147483648, foo(2147483647, 1))
        self.assertEqual(2147483647, foo(-2147483648, -1))

    def test_int_sub(self):
        def foo(a: int32, b: int32) -> int32:
            return a - b

        foo = self.helper(foo)

        self.assertEqual(-1, foo(0, 1))
        self.assertEqual(-2, foo(-1, 1))
        self.assertEqual(0, foo(1, 1))
        self.assertEqual(3, foo(-5, -8))
        self.assertEqual(-2, foo(5, 7))

    def test_int_mul(self):
        def foo(a: int32, b: int32) -> int32:
            return a * b

        foo = self.helper(foo)
        self.assertEqual(0, foo(0, 1))
        self.assertEqual(-1, foo(-1, 1))
        self.assertEqual(1, foo(1, 1))
        self.assertEqual(40, foo(-5, -8))
        self.assertEqual(35, foo(5, 7))

    def test_int_div(self):
        # numpy int32/int32 = float 64
        def foo(a: int32, b: int32) -> float64:
            return a / b

        foo = self.helper(foo)
        self.assertEqual(0 / 1, foo(0, 1))
        self.assertEqual(-1 / 1, foo(-1, 1))
        self.assertEqual(1 / 1, foo(1, 1))
        self.assertEqual(-5 / -8, foo(-5, -8))
        self.assertEqual(5 / 7, foo(5, 7))

    def test_int_rem(self):
        def foo(a: int32, b: int32) -> int32:
            return a % b

        foo = self.helper(foo)
        self.assertEqual(-10 % 3, foo(-10, 3))
        self.assertEqual(0 % 1, foo(0, 1))
        self.assertEqual(-1 % 1, foo(-1, 1))
        self.assertEqual(1 % 1, foo(1, 1))
        self.assertEqual(-5 % -8, foo(-5, -8))
        self.assertEqual(5 % -8, foo(5, -8))
        self.assertEqual(5 % 7, foo(5, 7))
        self.assertEqual(-5 % 7, foo(-5, 7))
        self.assertEqual(18328 % 32202, foo(18328, 32202))
        self.assertEqual(32202 % 18328, foo(32202, 18328))
        self.assertEqual(-18328 % 32202, foo(-18328, 32202))
        self.assertEqual(18328 % -32202, foo(18328, -32202))

    def test_int_floordiv(self):
        def foo(a: int32, b: int32) -> int32:
            return a // b

        foo = self.helper(foo)
        self.assertEqual(0 // 1, foo(0, 1))
        self.assertEqual(-1 // 1, foo(-1, 1))
        self.assertEqual(1 // 1, foo(1, 1))
        self.assertEqual(-5 // -8, foo(-5, -8))
        self.assertEqual(5 // 7, foo(5, 7))
        self.assertEqual(-3, foo(5, -2))
        self.assertEqual(2, foo(5, 2))

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

        self.assertAlmostEquals(1.213, foo(0, 1.213))
        self.assertAlmostEquals(1.213, foo(0, 1.213))
        self.assertAlmostEquals(1.313, foo(0.1, 1.213))
        self.assertAlmostEquals(0, foo(-1.1, 1.1))
        self.assertAlmostEquals(-13.9001, foo(-5.9, -8.0001))
        self.assertAlmostEquals(5.34 + 7.287953, foo(5.34, 7.287953))

    def test_float_sub(self):
        def foo(a: float64, b: float64) -> float64:
            return a - b

        foo = self.helper(foo)

        self.assertAlmostEquals(-1.213, foo(0, 1.213))
        self.assertAlmostEquals(-1.213, foo(0, 1.213))
        self.assertAlmostEquals(0.1 - 1.213, foo(0.1, 1.213))
        self.assertAlmostEquals(-1.1 - 1.1, foo(-1.1, 1.1))
        self.assertAlmostEquals(-5.9 - (-8.0001), foo(-5.9, -8.0001))
        self.assertAlmostEquals(5.34 - 7.287953, foo(5.34, 7.287953))

    def test_float_mul(self):
        def foo(a: float64, b: float64) -> float64:
            return a * b

        foo = self.helper(foo)
        self.assertAlmostEquals(0 * 1.213, foo(0, 1.213))
        self.assertAlmostEquals(0 * 1.213, foo(0, 1.213))
        self.assertAlmostEquals(0.1 * 1.213, foo(0.1, 1.213))
        self.assertAlmostEquals(-1.1 * 1.1, foo(-1.1, 1.1))
        self.assertAlmostEquals(-5.9 * (-8.0001), foo(-5.9, -8.0001))
        self.assertAlmostEquals(5.34 * 7.287953, foo(5.34, 7.287953))

    def test_float_div(self):
        # numpy int32/int32 = float 64
        def foo(a: float64, b: float64) -> float64:
            return a / b

        foo = self.helper(foo)
        self.assertAlmostEquals(0 / 1.213, foo(0, 1.213))
        self.assertAlmostEquals(0 / 1.213, foo(0, 1.213))
        self.assertAlmostEquals(0.1 / 1.213, foo(0.1, 1.213))
        self.assertAlmostEquals(-1.1 / 1.1, foo(-1.1, 1.1))
        self.assertAlmostEquals(-5.9 / (-8.0001), foo(-5.9, -8.0001))
        self.assertAlmostEquals(5.34 / 7.287953, foo(5.34, 7.287953))

    def test_float_rem(self):
        def foo(a: float64, b: float64) -> float64:
            return a % b

        foo = self.helper(foo)
        self.assertEqual(10 % -3, foo(10, -3))
        self.assertEqual(-10 % 3, foo(-10, 3))
        self.assertEqual(0 % 1.1, foo(0, 1.1))
        self.assertEqual(-1.1 % 1.2, foo(-1.1, 1.2))
        self.assertEqual(1 % 1, foo(1, 1))
        self.assertEqual(-5 % -8, foo(-5, -8))
        self.assertEqual(5 % -8, foo(5, -8))
        self.assertEqual(5.5 % -8.3, foo(5.5, -8.3))
        self.assertEqual(5 % 7, foo(5, 7))
        self.assertEqual(-5 % 7, foo(-5, 7))
        self.assertEqual(-5.34 % 7.68, foo(-5.34, 7.68))
        self.assertEqual(18328 % 32202, foo(18328, 32202))
        self.assertEqual(32202 % 18328, foo(32202, 18328))
        self.assertEqual(-18328 % 32202, foo(-18328, 32202))
        self.assertEqual(-18328.3534 % 32202.3534, foo(-18328.3534, 32202.3534))
        self.assertEqual(18328 % -32202, foo(18328, -32202))

    def test_float_floordiv(self):
        # numpy float64/float64 = float 64
        def foo(a: float64, b: float64) -> float64:
            return a // b

        foo = self.helper(foo)
        self.assertAlmostEquals(0 // 1.213, foo(0, 1.213))
        self.assertAlmostEquals(0 // 1.213, foo(0, 1.213))
        self.assertAlmostEquals(0.1 // 1.213, foo(0.1, 1.213))
        self.assertAlmostEquals(-1.1 // 1.1, foo(-1.1, 1.1))
        self.assertAlmostEquals(-5.9 // (-8.0001), foo(-5.9, -8.0001))
        self.assertAlmostEquals(5.34 // 7.287953, foo(5.34, 7.287953))
        self.assertEqual(-3, foo(5, -2))
        self.assertEqual(2, foo(5, 2))

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


class TestMLIRMixedArithmeticOp(unittest.TestCase):
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

    def mixed_run_helper(self, int_a, float_b, f):
        self.assertAlmostEquals(f(int_a, float_b), self.foo1(int_a, float_b))
        self.assertAlmostEquals(f(float_b, int_a), self.foo2(float_b, int_a))
        self.assertAlmostEquals(f(float_b, int_a), self.foo3(float_b, int_a), places=3)

    def test_Mixed_add(self):
        def foo1(a: int32, b: float64) -> float64:
            return a + b

        def foo2(a: float64, b: int32) -> float64:
            return a + b

        def foo3(a: float32, b: int32) -> float32:
            return a + b

        self.foo1 = self.helper(foo1)
        self.foo2 = self.helper(foo2)
        self.foo3 = self.helper(foo3)

        self.mixed_run_helper(0, 1.213, lambda a, b: a + b)
        self.mixed_run_helper(-1, 1.1, lambda a, b: a + b)
        self.mixed_run_helper(-5, -8.0001, lambda a, b: a + b)
        self.mixed_run_helper(5, 7.287953, lambda a, b: a + b)

    def test_Mixed_sub(self):
        def foo1(a: int32, b: float64) -> float64:
            return a - b

        def foo2(a: float64, b: int32) -> float64:
            return a - b

        def foo3(a: float32, b: int32) -> float32:
            return a - b

        self.foo1 = self.helper(foo1)
        self.foo2 = self.helper(foo2)
        self.foo3 = self.helper(foo3)

        self.mixed_run_helper(0, 1.213, lambda a, b: a - b)
        self.mixed_run_helper(-1, 1.1, lambda a, b: a - b)
        self.mixed_run_helper(-5, -8.0001, lambda a, b: a - b)
        self.mixed_run_helper(5, 7.287953, lambda a, b: a - b)

    def test_Mixed_mul(self):
        def foo1(a: int32, b: float64) -> float64:
            return a * b

        def foo2(a: float64, b: int32) -> float64:
            return a * b

        def foo3(a: float32, b: int32) -> float32:
            return a * b

        self.foo1 = self.helper(foo1)
        self.foo2 = self.helper(foo2)
        self.foo3 = self.helper(foo3)

        self.mixed_run_helper(0, 1.213, lambda a, b: a * b)
        self.mixed_run_helper(-1, 1.1, lambda a, b: a * b)
        self.mixed_run_helper(-5, -8.0001, lambda a, b: a * b)
        self.mixed_run_helper(5, 7.287953, lambda a, b: a * b)

    def test_Mixed_div(self):
        def foo1(a: int32, b: float64) -> float64:
            return a / b

        def foo2(a: float64, b: int32) -> float64:
            return a / b

        # numpy float32/int32 = float 64
        def foo3(a: float32, b: int32) -> float64:
            return a / b

        self.foo1 = self.helper(foo1)
        self.foo2 = self.helper(foo2)
        self.foo3 = self.helper(foo3)

        self.mixed_run_helper(6, 1.213, lambda a, b: a / b)
        self.mixed_run_helper(-1, 1.1, lambda a, b: a / b)
        self.mixed_run_helper(-5, -8.0001, lambda a, b: a / b)
        self.mixed_run_helper(5, 7.287953, lambda a, b: a / b)

    def test_Mixed_rem(self):
        def foo1(a: int32, b: float64) -> float64:
            return a % b

        def foo2(a: float64, b: int32) -> float64:
            return a % b

        def foo3(a: float32, b: int32) -> float64:
            return a % b

        self.foo1 = self.helper(foo1)
        self.foo2 = self.helper(foo2)
        self.foo3 = self.helper(foo3)
        self.mixed_run_helper(10, -3, lambda a, b: a % b)
        self.mixed_run_helper(-10, 3, lambda a, b: a % b)
        self.mixed_run_helper(1, 1, lambda a, b: a % b)
        self.mixed_run_helper(-5, -8, lambda a, b: a % b)
        self.mixed_run_helper(5, -8, lambda a, b: a % b)
        self.mixed_run_helper(5, -8.3, lambda a, b: a % b)
        self.mixed_run_helper(5, 7, lambda a, b: a % b)
        self.mixed_run_helper(-5, 7, lambda a, b: a % b)
        self.mixed_run_helper(5, 7, lambda a, b: a % b)
        self.mixed_run_helper(-5, 7.68, lambda a, b: a % b)
        self.mixed_run_helper(18328, 32202, lambda a, b: a % b)
        self.mixed_run_helper(32202, 18328, lambda a, b: a % b)
        self.mixed_run_helper(-18328, 32202, lambda a, b: a % b)
        self.mixed_run_helper(-18328, 32202.3534, lambda a, b: a % b)
        self.mixed_run_helper(18328, -32202, lambda a, b: a % b)

    def test_Mixed_floordiv(self):
        # numpy float64/float64 = float 64
        def foo1(a: int32, b: float64) -> float64:
            return a // b

        def foo2(a: float64, b: int32) -> float64:
            return a // b

        def foo3(a: float32, b: int32) -> float64:
            return a // b

        self.foo1 = self.helper(foo1)
        self.foo2 = self.helper(foo2)
        self.foo3 = self.helper(foo3)

        self.mixed_run_helper(6, 1.213, lambda a, b: a // b)
        self.mixed_run_helper(-1, 1.1, lambda a, b: a // b)
        self.mixed_run_helper(-5, -8.0001, lambda a, b: a // b)
        self.mixed_run_helper(5, 7.287953, lambda a, b: a // b)
        self.mixed_run_helper(5, -2, lambda a, b: a // b)
        self.mixed_run_helper(5, 2, lambda a, b: a // b)

    def test_Mixed_min(self):
        def foo(a: float64, b: float64) -> float64:
            return min(a, b)
        # todo not supported yet
        # foo = self.helper(foo)

    def test_Mixed_max(self):
        def foo(a: float64, b: float64) -> float64:
            return max(a, b)
        # todo not supported yet
        # foo = self.helper(foo)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
