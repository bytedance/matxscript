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

import sympy

from matx.kernel.compile_linalg import compile_linalg
from matx.kernel.kernel_parser import KernelParser
from matx.kernel.typing import *


class TestSingleReturnParser(unittest.TestCase):

    def test_no_op(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[M, N]:
            return a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        rt = np.array([[0, 0], [0, 0]], dtype=np.int32)
        f = compile_linalg(p)
        f(a, rt=rt)
        np.testing.assert_equal(rt, foo(a))
        # todo check ir structure

    def test_one_bin_op(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N]) -> int32[M, N]:
            return a + b

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, rt=rt)
        np.testing.assert_equal(rt, foo(a, b))

    def test_multiple_bin_op(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N], c: int32[M, N]) -> int32[M, N]:
            return a + b - c * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
        c = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.int32)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))

    def test_multiple_bin_op_with_parentheses(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N], c: int32[M, N]) -> int32[M, N]:
            return a + (b - c) * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
        c = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.int32)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))

    def test_multiple_bin_op_with_broadcast(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N], c: int32[N]) -> int32[M, N]:
            return a + (b - c) * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
        c = np.array([13, 14, 15], dtype=np.int32)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))

    def test_multiple_bin_op_with_more_dimension(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)
        K = sympy.Symbol('K', positive=True)

        def foo(a: int32[K, M, N], b: int32[K, M, N], c: int32[K, M, N]) -> int32[K, M, N]:
            return a + (b - c) * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        b = np.arange(24, 48, dtype=np.int32).reshape(2, 3, 4)
        c = np.arange(48, 72, dtype=np.int32).reshape(2, 3, 4)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))

    def test_multiple_bin_op_with_different_type(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int64[M, N], c: float32[M, N]) -> float64[M, N]:
            return (a + b) * c

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.arange(12, dtype=np.int32).reshape(3, 4)
        b = np.arange(12, 24, dtype=np.int64).reshape(3, 4)
        c = np.arange(24, 36, dtype=np.float32).reshape(3, 4)
        rt = np.zeros(a.shape, dtype=np.float64)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))

    def test_multiple_bin_op_with_scalar_and_const(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int64[M, N], c: float32) -> float64[M, N]:
            return ((a + b) * c) + 1

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 30, "linalg_code", "=" * 30, sep="")
        print()
        print(p.linalg_code())
        print()
        print("=" * 30, "compile and run", "=" * 30, sep="")
        print()
        a = np.arange(12, dtype=np.int32).reshape(3, 4)
        b = np.arange(12, 24, dtype=np.int64).reshape(3, 4)
        c = np.float32(3)
        rt = np.zeros(a.shape, dtype=np.float64)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
