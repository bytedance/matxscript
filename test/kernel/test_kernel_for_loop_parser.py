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

from matx.kernel.kernel_parser import KernelParser
from matx.kernel.typing import int32, float32


class TestSingleReturnParser(unittest.TestCase):
    def test_scalar_op1(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[M, N]:
            for i in range(M):
                for j in range(N):
                    k: int32 = i + j
            return a

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_scalar_op2(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[M, N]:
            k: int32 = 0
            for i in range(M):
                for j in range(N):
                    k = i + j
            return a

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_idx_operation1(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[M, N]:
            k: int32 = 0
            for i in range(M):
                for j in range(N):
                    k = a[(i + 1) / 2, j] + j
            return a

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_idx_operation2(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[M, N]:
            k: int32 = 0
            for i in range(M):
                for j in range(N):
                    a[(i + 1) / 2, j] = i + j
            return a

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_scalar_rhs_nd1(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N]) -> int32[M, N]:
            k: int32 = 0
            for i in range(M):
                for j in range(N):
                    k = a[i, j] + b[i, j]
            return a

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_scalar_rhs_nd2(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N]) -> int32[M, N]:
            for i in range(M):
                for j in range(N):
                    k: int32 = a[i, j] + 1
            return a

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_scalar_both_nd(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N]) -> int32[M, N]:
            for i in range(M):
                for j in range(N):
                    b[i, j] = a[i, j] + 1
            return b

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_scalar_both_nd_different_type(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: float32[M, N]) -> int32[M, N]:
            for i in range(M):
                for j in range(N):
                    b[i, j] = a[i, j] + b[i, j]
            return b

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_scalar_different_type_many_nd(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: float32[M, N], c: int32[N], d: float32[M]) -> int32[M, N]:
            for i in range(M):
                for j in range(N):
                    b[i, j] = (a[i, j] + c[j]) / (b[i, j] * d[i])
            return b

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure

    def test_scalar_op_on_idx(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: float32[2 * M, N], c: int32[N], d: float32[M]) -> int32[M, N]:
            for i in range(M):
                for j in range(N):
                    b[i + 1, j] = (a[i, j] + c[j]) / (b[i, j] * d[i])
            return b

        p = KernelParser(foo)
        p.parse()
        # todo check ir structure
