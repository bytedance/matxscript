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
from matx.kernel.compile_linalg import compile_linalg, run
from matx.kernel.typing import int32, int64, float32


class TestSingleReturnParser(unittest.TestCase):

    def test_no_op(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N]) -> int32[M, N]:
            return a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        run(p, np.array([[1, 2], [3, 4]], dtype=np.int32), rt=np.array([[0, 0], [0, 0]], dtype=np.int32))
        # todo check ir structure

    def test_one_bin_op(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N]) -> int32[M, N]:
            return a + b

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        compile_linalg(p.main_node_ir)
        # todo check ir structure

    def test_multiple_bin_op(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N], c: int32[M, N]) -> int32[M, N]:
            return a + b - c * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        compile_linalg(p.main_node_ir)
        # todo check ir structure

    def test_multiple_bin_op_with_parentheses(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N], c: int32[M, N]) -> int32[M, N]:
            return a + (b - c) * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        compile_linalg(p.main_node_ir)
        # todo check ir structure

    def test_multiple_bin_op_with_broadcast(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int32[M, N], c: int32[N]) -> int32[M, N]:
            return a + (b - c) * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        compile_linalg(p.main_node_ir)
        # todo check ir structure

    def test_multiple_bin_op_with_more_dimension(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)
        K = sympy.Symbol('K', positive=True)

        def foo(a: int32[K, M, N], b: int32[K, M, N], c: int32[K, M, N]) -> int32[K, M, N]:
            return a + (b - c) * a

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        compile_linalg(p.main_node_ir)
        # todo check ir structure

    def test_multiple_bin_op_with_different_type(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int64[M, N], c: float32[M, N]) -> float32[M, N]:
            return (a + b) * c

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        compile_linalg(p.main_node_ir)
        # todo check ir structure

    def test_multiple_bin_op_with_scalar_and_const(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int64[M, N], c: float32) -> float32[M, N]:
            return ((a + b) * c) + 1

        p = KernelParser(foo)
        p.parse()
        print()
        print("=" * 100)
        print()
        print(p.linalg_code())
        compile_linalg(p.main_node_ir)
        # todo check ir structure
