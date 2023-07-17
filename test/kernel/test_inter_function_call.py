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
import numpy as np
import sympy

from matx.kernel.kernel_parser import KernelParser
from matx.kernel.compile_linalg import compile_linalg
from matx.kernel.typing import int32, int64, float32


class TestSingleReturnParser(unittest.TestCase):

    def test_two_op(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def boo(a: int32[M, N], b: int32[M, N]) -> int32[M, N]:
            return a - b

        def foo(a: int32[M, N], b: int32[M, N], c: int32[M, N]) -> int32[M, N]:
            return a + boo(b, c)

        # todo check ir structure
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
        print(a.shape)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))

    def test_two_op_diff_shape_symbol(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)
        K = sympy.Symbol('K', positive=True)

        def boo(a: int32[K, N], b: int32[K, N]) -> int32[K, N]:
            return a - b

        def foo(a: int32[M, N], b: int32[M, N], c: int32[M, N]) -> int32[M, N]:
            return a + boo(b, c)

        # todo check ir structure
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
        print(a.shape)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))


    def test_two_op_diff_shape_symbol2(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def boo(a: int32[N, M], b: int32[N, M]) -> int32[N, M]:
            return a - b

        def foo(a: int32[M, N], b: int32[M, N], c: int32[M, N]) -> int32[M, N]:
            return a + boo(b, c)

        # todo check ir structure
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
        print(a.shape)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))


    def test_two_op_multi_call(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)
        K = sympy.Symbol('K', positive=True)
        P = sympy.Symbol('P', positive=True)
        Q = sympy.Symbol('Q', positive=True)

        def boo1(a: int32[M, K], b: int32[M, K]) -> int32[M, K]:
            return a - b

        def boo2(c: int32[P, Q], d: int32[P, Q]) -> int32[P, Q]:
            return boo1(c, d) + c + d #+ 1

        def foo(e: int32[M, N], f: int32[M, N], g: int32[M, N]) -> int32[M, N]:
            return boo2(e, f) #+ e + boo1(f, g)

        # todo check ir structure
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
        print(a.shape)
        rt = np.zeros(a.shape, dtype=np.int32)
        f = compile_linalg(p)
        f(a, b, c, rt=rt)
        np.testing.assert_equal(rt, foo(a, b, c))