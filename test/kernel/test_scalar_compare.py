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


class TestMLIRIntComparsion(unittest.TestCase):
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

    def test_int_eq(self):
        def foo(a: int32, b: int32) -> boolean:
            return a == b

        foo = self.helper(foo)

        self.assertTrue(foo(-2, -2))
        self.assertTrue(foo(-1, -1))
        self.assertTrue(foo(-1000, -1000))
        self.assertTrue(foo(0, 0))
        self.assertTrue(foo(1, 1))
        self.assertTrue(foo(2, 2))

        self.assertFalse(foo(-100, -2))
        self.assertFalse(foo(-2, -100))
        self.assertFalse(foo(1, -1))
        self.assertFalse(foo(1, 0))
        self.assertFalse(foo(0, 1))
        self.assertFalse(foo(2, 200))

    def test_int_ne(self):
        def foo(a: int32, b: int32) -> boolean:
            return a != b

        foo = self.helper(foo)

        self.assertFalse(foo(-2, -2))
        self.assertFalse(foo(-1, -1))
        self.assertFalse(foo(-1000, -1000))
        self.assertFalse(foo(0, 0))
        self.assertFalse(foo(1, 1))
        self.assertFalse(foo(2, 2))

        self.assertTrue(foo(-100, -2))
        self.assertTrue(foo(-2, -100))
        self.assertTrue(foo(1, -1))
        self.assertTrue(foo(1, 0))
        self.assertTrue(foo(0, 1))
        self.assertTrue(foo(2, 200))

    def test_int_lt(self):
        def foo(a: int32, b: int32) -> boolean:
            return a < b

        foo = self.helper(foo)

        self.assertFalse(foo(-2, -2))
        self.assertFalse(foo(-1, -1))
        self.assertFalse(foo(-1000, -1000))
        self.assertFalse(foo(0, 0))
        self.assertFalse(foo(1, 1))
        self.assertFalse(foo(2, 2))

        self.assertTrue(foo(-100, -2))
        self.assertTrue(foo(-1, 0))
        self.assertTrue(foo(0, 1))
        self.assertTrue(foo(-1, 1))
        self.assertTrue(foo(1, 3))
        self.assertTrue(foo(2, 200))

        self.assertFalse(foo(-2, -100))
        self.assertFalse(foo(0, -1))
        self.assertFalse(foo(1, 0))
        self.assertFalse(foo(1, -1))
        self.assertFalse(foo(3, 1))
        self.assertFalse(foo(200, 2))

    def test_int_le(self):
        def foo(a: int32, b: int32) -> boolean:
            return a <= b

        foo = self.helper(foo)
        self.assertTrue(foo(-2, -2))
        self.assertTrue(foo(-1, -1))
        self.assertTrue(foo(-1000, -1000))
        self.assertTrue(foo(0, 0))
        self.assertTrue(foo(1, 1))
        self.assertTrue(foo(2, 2))

        self.assertTrue(foo(-100, -2))
        self.assertTrue(foo(-1, 0))
        self.assertTrue(foo(0, 1))
        self.assertTrue(foo(-1, 1))
        self.assertTrue(foo(1, 3))
        self.assertTrue(foo(2, 200))

        self.assertFalse(foo(-2, -100))
        self.assertFalse(foo(0, -1))
        self.assertFalse(foo(1, 0))
        self.assertFalse(foo(1, -1))
        self.assertFalse(foo(3, 1))
        self.assertFalse(foo(200, 2))

    def test_int_gt(self):
        def foo(a: int32, b: int32) -> boolean:
            return a > b

        foo = self.helper(foo)
        self.assertFalse(foo(-2, -2))
        self.assertFalse(foo(-1, -1))
        self.assertFalse(foo(-1000, -1000))
        self.assertFalse(foo(0, 0))
        self.assertFalse(foo(1, 1))
        self.assertFalse(foo(2, 2))

        self.assertFalse(foo(-100, -2))
        self.assertFalse(foo(-1, 0))
        self.assertFalse(foo(0, 1))
        self.assertFalse(foo(-1, 1))
        self.assertFalse(foo(1, 3))
        self.assertFalse(foo(2, 200))

        self.assertTrue(foo(-2, -100))
        self.assertTrue(foo(0, -1))
        self.assertTrue(foo(1, 0))
        self.assertTrue(foo(1, -1))
        self.assertTrue(foo(3, 1))
        self.assertTrue(foo(200, 2))

    def test_int_ge(self):
        def foo(a: int32, b: int32) -> boolean:
            return a >= b

        foo = self.helper(foo)
        self.assertTrue(foo(-2, -2))
        self.assertTrue(foo(-1, -1))
        self.assertTrue(foo(-1000, -1000))
        self.assertTrue(foo(0, 0))
        self.assertTrue(foo(1, 1))
        self.assertTrue(foo(2, 2))

        self.assertFalse(foo(-100, -2))
        self.assertFalse(foo(-1, 0))
        self.assertFalse(foo(0, 1))
        self.assertFalse(foo(-1, 1))
        self.assertFalse(foo(1, 3))
        self.assertFalse(foo(2, 200))

        self.assertTrue(foo(-2, -100))
        self.assertTrue(foo(0, -1))
        self.assertTrue(foo(1, 0))
        self.assertTrue(foo(1, -1))
        self.assertTrue(foo(3, 1))
        self.assertTrue(foo(200, 2))


class TestMLIRFloatComparsion(unittest.TestCase):
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

    def test_float_eq(self):
        def foo(a: float32, b: float32) -> boolean:
            return a == b

        foo = self.helper(foo)

        self.assertTrue(foo(-2.1, -2.1))
        self.assertTrue(foo(-1.3, -1.3))
        self.assertTrue(foo(-1000.233, -1000.233))
        self.assertTrue(foo(0, 0))
        self.assertTrue(foo(0.00000001, 0.00000001))
        self.assertTrue(foo(2.3, 2.3))

        self.assertFalse(foo(-100.231, -2.455))
        self.assertFalse(foo(-2, -100.1313))
        self.assertFalse(foo(1.1, -1.1))
        self.assertFalse(foo(1.2, 0))
        self.assertFalse(foo(1, 1.0000001))
        self.assertFalse(foo(1, 0.0000009))
        self.assertFalse(foo(2.9, 200.1))

        # test orderness
        self.assertFalse(foo(-2, float("nan")))
        self.assertFalse(foo(float("nan"), -2.5))
        self.assertFalse(foo(-100, float("nan")))
        self.assertFalse(foo(float("nan"), -2))
        self.assertFalse(foo(-2, float("nan")))
        self.assertFalse(foo(float("nan"), -2.00001))

    def test_float_ne(self):
        def foo(a: float32, b: float32) -> boolean:
            return a != b

        foo = self.helper(foo)

        self.assertFalse(foo(-2.1, -2.1))
        self.assertFalse(foo(-1.3, -1.3))
        self.assertFalse(foo(-1000.233, -1000.233))
        self.assertFalse(foo(0, 0))
        self.assertFalse(foo(0.00000001, 0.00000001))
        self.assertFalse(foo(2.3, 2.3))

        self.assertTrue(foo(-100.231, -2.455))
        self.assertTrue(foo(-2, -100.1313))
        self.assertTrue(foo(1.1, -1.1))
        self.assertTrue(foo(1.2, 0))
        self.assertTrue(foo(1, 1.0000001))
        self.assertTrue(foo(1, 0.0000009))
        self.assertTrue(foo(2.9, 200.1))

        # test orderness
        self.assertFalse(foo(-2, float("nan")))
        self.assertFalse(foo(float("nan"), -2.5))
        self.assertFalse(foo(-100, float("nan")))
        self.assertFalse(foo(float("nan"), -2))
        self.assertFalse(foo(-2, float("nan")))
        self.assertFalse(foo(float("nan"), -2.00001))

    def test_float_lt(self):
        def foo_32(a: float32, b: float32) -> boolean:
            return a < b

        def foo_64(a: float64, b: float64) -> boolean:
            return a < b

        foo_32 = self.helper(foo_32)
        foo_64 = self.helper(foo_64)

        self.assertFalse(foo_32(-2, -2))
        self.assertFalse(foo_32(-2.5, -2.5))
        self.assertFalse(foo_32(-1, -1))
        self.assertFalse(foo_32(-1.8, -1.8))
        self.assertFalse(foo_32(-1000, -1000))
        self.assertFalse(foo_32(-1000.0, -1000.0))
        self.assertFalse(foo_32(0.1, 0.1))
        self.assertFalse(foo_32(0.0, 0.0))
        self.assertFalse(foo_32(1.00000001, 1.00000001))
        self.assertFalse(foo_32(2, 2))
        self.assertFalse(foo_32(2.6, 2.6))

        self.assertTrue(foo_32(-100, -2))
        self.assertTrue(foo_32(-2.00001, -2))
        self.assertTrue(foo_32(-0.0001, 0))
        self.assertTrue(foo_32(-1.346456, 0))
        self.assertTrue(foo_32(0, 1.7823))
        self.assertTrue(foo_32(0, 0.000001))
        self.assertTrue(foo_32(-1, 1))
        self.assertTrue(foo_32(1.9999, 2.000))
        self.assertTrue(foo_32(2.1, 200.96))
        # due to floating point precision, use float64
        # self.assertTrue(foo_32(1, 1.000000000001))
        # self.assertTrue(foo_32(1.00000000000001, 1.000000000001))
        self.assertTrue(foo_64(1, 1.000000000001))
        self.assertTrue(foo_64(1.00000000000001, 1.000000000001))

        self.assertFalse(foo_32(-2, -100))
        self.assertFalse(foo_32(-2, -2.00001))
        self.assertFalse(foo_32(0, -0.0001))
        self.assertFalse(foo_32(0, -1.346456))
        self.assertFalse(foo_32(1.7823, 0))
        self.assertFalse(foo_32(0.000001, 0))
        self.assertFalse(foo_32(1, -1))
        self.assertFalse(foo_32(2.000, 1.9999))
        self.assertFalse(foo_32(200.96, 2.1))
        # due to floating point precision, use float64
        # self.assertFalse(foo_32(1.000000000001, 1))
        # self.assertFalse(foo_32(1.000000000001, 1.00000000000001))
        self.assertFalse(foo_64(1.000000000001, 1))
        self.assertFalse(foo_64(1.000000000001, 1.00000000000001))

        # test orderness
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.5))
        self.assertFalse(foo_32(-100, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2))
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.00001))

    def test_float_le(self):
        def foo_32(a: float32, b: float32) -> boolean:
            return a <= b

        def foo_64(a: float64, b: float64) -> boolean:
            return a <= b

        foo_32 = self.helper(foo_32)
        foo_64 = self.helper(foo_64)

        self.assertTrue(foo_32(-2, -2))
        self.assertTrue(foo_32(-2.5, -2.5))
        self.assertTrue(foo_32(-1, -1))
        self.assertTrue(foo_32(-1.8, -1.8))
        self.assertTrue(foo_32(-1000, -1000))
        self.assertTrue(foo_32(-1000.0, -1000.0))
        self.assertTrue(foo_32(0.1, 0.1))
        self.assertTrue(foo_32(0.0, 0.0))
        self.assertTrue(foo_32(1.00000001, 1.00000001))
        self.assertTrue(foo_32(2, 2))
        self.assertTrue(foo_32(2.6, 2.6))

        self.assertTrue(foo_32(-100, -2))
        self.assertTrue(foo_32(-2.00001, -2))
        self.assertTrue(foo_32(-0.0001, 0))
        self.assertTrue(foo_32(-1.346456, 0))
        self.assertTrue(foo_32(0, 1.7823))
        self.assertTrue(foo_32(0, 0.000001))
        self.assertTrue(foo_32(-1, 1))
        self.assertTrue(foo_32(1.9999, 2.000))
        self.assertTrue(foo_32(2.1, 200.96))
        # due to floating point precision, use float64
        # self.assertTrue(foo_32(1, 1.000000000001))
        # self.assertTrue(foo_32(1.00000000000001, 1.000000000001))
        self.assertTrue(foo_64(1, 1.000000000001))
        self.assertTrue(foo_64(1.00000000000001, 1.000000000001))

        self.assertFalse(foo_32(-2, -100))
        self.assertFalse(foo_32(-2, -2.00001))
        self.assertFalse(foo_32(0, -0.0001))
        self.assertFalse(foo_32(0, -1.346456))
        self.assertFalse(foo_32(1.7823, 0))
        self.assertFalse(foo_32(0.000001, 0))
        self.assertFalse(foo_32(1, -1))
        self.assertFalse(foo_32(2.000, 1.9999))
        self.assertFalse(foo_32(200.96, 2.1))
        # due to floating point precision, use float64
        # self.assertFalse(foo_32(1.000000000001, 1))
        # self.assertFalse(foo_32(1.000000000001, 1.00000000000001))
        self.assertFalse(foo_64(1.000000000001, 1))
        self.assertFalse(foo_64(1.000000000001, 1.00000000000001))

        # test orderness
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.5))
        self.assertFalse(foo_32(-100, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2))
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.00001))

    def test_float_gt(self):
        def foo_32(a: float32, b: float32) -> boolean:
            return a > b

        def foo_64(a: float64, b: float64) -> boolean:
            return a > b

        foo_32 = self.helper(foo_32)
        foo_64 = self.helper(foo_64)

        self.assertFalse(foo_32(-2, -2))
        self.assertFalse(foo_32(-2.5, -2.5))
        self.assertFalse(foo_32(-1, -1))
        self.assertFalse(foo_32(-1.8, -1.8))
        self.assertFalse(foo_32(-1000, -1000))
        self.assertFalse(foo_32(-1000.0, -1000.0))
        self.assertFalse(foo_32(0.1, 0.1))
        self.assertFalse(foo_32(0.0, 0.0))
        self.assertFalse(foo_32(1.00000001, 1.00000001))
        self.assertFalse(foo_32(2, 2))
        self.assertFalse(foo_32(2.6, 2.6))

        self.assertFalse(foo_32(-100, -2))
        self.assertFalse(foo_32(-2.00001, -2))
        self.assertFalse(foo_32(-0.0001, 0))
        self.assertFalse(foo_32(-1.346456, 0))
        self.assertFalse(foo_32(0, 1.7823))
        self.assertFalse(foo_32(0, 0.000001))
        self.assertFalse(foo_32(-1, 1))
        self.assertFalse(foo_32(1.9999, 2.000))
        self.assertFalse(foo_32(2.1, 200.96))
        # due to floating point precision, use float64
        # self.assertFalse(foo_32(1, 1.000000000001))
        # self.assertFalse(foo_32(1.00000000000001, 1.000000000001))
        self.assertFalse(foo_64(1, 1.000000000001))
        self.assertFalse(foo_64(1.00000000000001, 1.000000000001))

        self.assertTrue(foo_32(-2, -100))
        self.assertTrue(foo_32(-2, -2.00001))
        self.assertTrue(foo_32(0, -0.0001))
        self.assertTrue(foo_32(0, -1.346456))
        self.assertTrue(foo_32(1.7823, 0))
        self.assertTrue(foo_32(0.000001, 0))
        self.assertTrue(foo_32(1, -1))
        self.assertTrue(foo_32(2.000, 1.9999))
        self.assertTrue(foo_32(200.96, 2.1))
        # due to floating point precision, use float64
        # self.assertTrue(foo_32(1.000000000001, 1))
        # self.assertTrue(foo_32(1.000000000001, 1.00000000000001))
        self.assertTrue(foo_64(1.000000000001, 1))
        self.assertTrue(foo_64(1.000000000001, 1.00000000000001))

        # test orderness
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.5))
        self.assertFalse(foo_32(-100, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2))
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.00001))

    def test_float_ge(self):
        def foo_32(a: float32, b: float32) -> boolean:
            return a >= b

        def foo_64(a: float64, b: float64) -> boolean:
            return a >= b

        foo_32 = self.helper(foo_32)
        foo_64 = self.helper(foo_64)

        self.assertTrue(foo_32(-2, -2))
        self.assertTrue(foo_32(-2.5, -2.5))
        self.assertTrue(foo_32(-1, -1))
        self.assertTrue(foo_32(-1.8, -1.8))
        self.assertTrue(foo_32(-1000, -1000))
        self.assertTrue(foo_32(-1000.0, -1000.0))
        self.assertTrue(foo_32(0.1, 0.1))
        self.assertTrue(foo_32(0.0, 0.0))
        self.assertTrue(foo_32(1.00000001, 1.00000001))
        self.assertTrue(foo_32(2, 2))
        self.assertTrue(foo_32(2.6, 2.6))

        self.assertFalse(foo_32(-100, -2))
        self.assertFalse(foo_32(-2.00001, -2))
        self.assertFalse(foo_32(-0.0001, 0))
        self.assertFalse(foo_32(-1.346456, 0))
        self.assertFalse(foo_32(0, 1.7823))
        self.assertFalse(foo_32(0, 0.000001))
        self.assertFalse(foo_32(-1, 1))
        self.assertFalse(foo_32(1.9999, 2.000))
        self.assertFalse(foo_32(2.1, 200.96))
        # due to floating point precision, use float64
        # self.assertFalse(foo_32(1, 1.000000000001))
        # self.assertFalse(foo_32(1.00000000000001, 1.000000000001))
        self.assertFalse(foo_64(1, 1.000000000001))
        self.assertFalse(foo_64(1.00000000000001, 1.000000000001))

        self.assertTrue(foo_32(-2, -100))
        self.assertTrue(foo_32(-2, -2.00001))
        self.assertTrue(foo_32(0, -0.0001))
        self.assertTrue(foo_32(0, -1.346456))
        self.assertTrue(foo_32(1.7823, 0))
        self.assertTrue(foo_32(0.000001, 0))
        self.assertTrue(foo_32(1, -1))
        self.assertTrue(foo_32(2.000, 1.9999))
        self.assertTrue(foo_32(200.96, 2.1))
        # due to floating point precision, use float64
        # self.assertTrue(foo_32(1.000000000001, 1))
        # self.assertTrue(foo_32(1.000000000001, 1.00000000000001))
        self.assertTrue(foo_64(1.000000000001, 1))
        self.assertTrue(foo_64(1.000000000001, 1.00000000000001))

        # test orderness
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.5))
        self.assertFalse(foo_32(-100, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2))
        self.assertFalse(foo_32(-2, float("nan")))
        self.assertFalse(foo_32(float("nan"), -2.00001))
