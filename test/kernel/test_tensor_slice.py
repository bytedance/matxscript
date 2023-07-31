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

from matx.kernel.compile_linalg import compile_linalg
from matx.kernel.kernel_parser import KernelParser
from matx.kernel.typing import int32


class TestTensorSliceParser(unittest.TestCase):

    def test_simple_scalar_return(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32:
            return a[0, 0]

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
        f = compile_linalg(p)
        np.testing.assert_equal(f(a), foo(a))

    def test_simple_load(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[M, N]:
            a[0, 0] = 1
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
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        f = compile_linalg(p)
        np.testing.assert_equal(f(a), foo(a))

    def test_simple_tensor_return(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[N]:
            return a[0]

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
        f = compile_linalg(p)
        np.testing.assert_equal(f(a), foo(a))

    def test_constant_slice_tensor_return(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[2, 2]:
            return a[:2, :2]

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
        foo(a)
        f = compile_linalg(p)
        np.testing.assert_equal(f(a), foo(a))

    def test_constant_slice_tensor_return2(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]):
            return a[:2, :2]

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
        foo(a)
        f = compile_linalg(p)
        np.testing.assert_equal(f(a), foo(a))


"""
    def test_constant_slice_tensor_return3(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32):
            return a[b:b + 1, b:b * 2]

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
        b = 1
        foo(a, 1)
        f = compile_linalg(p)
        np.testing.assert_equal(f(a), foo(a, 1))
"""
