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

import itertools
import unittest

from matx.kernel.compile_linalg import compile_linalg
from matx.kernel.kernel_parser import KernelParser
from matx.kernel.typing import int32, float32, float64


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

    def test_int_assign(self):
        def foo(a: int32, b: int32) -> int32:
            c: int32 = a + b
            return a - c

        k_foo = self.helper(foo)
        for x, y in itertools.product([-50, -1, 0, 6, 32], repeat=2):
            self.assertEqual(foo(x, y), k_foo(x, y))

    def test_mixed_assign(self):
        def foo(a: int32, b: float32) -> float64:
            c: float32 = a + b
            return a - c

        k_foo = self.helper(foo)
        for x, y in itertools.product([-50, -1, 0, 6, 32], repeat=2):
            self.assertEqual(foo(x, y), k_foo(x, y))

    def test_int_reassign1(self):
        def foo(a: int32, b: int32, c: int32) -> int32:
            c1: int32 = b * c
            c1: int32 = 1 + c1
            return a + b - c1

        k_foo = self.helper(foo)
        for x, y, z in itertools.product([-50, -1, 0, 6, 32], repeat=3):
            self.assertEqual(foo(x, y, z), k_foo(x, y, z))

    def test_int_reassign2(self):
        def foo(a: int32, b: int32, c: int32) -> int32:
            c1: int32 = b * c
            c1 = 1 + c1
            return a + b - c1

        k_foo = self.helper(foo)
        for x, y, z in itertools.product([-50, -1, 0, 6, 32], repeat=3):
            self.assertEqual(foo(x, y, z), k_foo(x, y, z))

    def test_int_reassign3(self):
        def foo(a: int32, b: int32, c: int32) -> int32:
            c1: int32 = b * c
            c2: int32 = 1 + c1
            return a + b - c2

        k_foo = self.helper(foo)
        for x, y, z in itertools.product([-50, -1, 0, 6, 32], repeat=3):
            self.assertEqual(foo(x, y, z), k_foo(x, y, z))


"""
    def test_scalar_if1(self):
        def foo(a: int32, b: int32, c: int32) -> int32:
            if a % 2 == 0:
                d: int32 = b + c
            else:
                d: int32 = b - c
            return d

        k_foo = self.helper(foo)
        for x, y, z in itertools.product([-50, -1, 0, 6, 32], repeat=3):
            self.assertEqual(foo(x, y, z), k_foo(x, y, z))

    def test_scalar_if2(self):
        def foo(a: int32, b: int32, c: int32) -> int32:
            if a % 2 == 0:
                i: int32 = a % 3
                d: int32 = b + c + i
            else:
                i: int32 = a % 4
                d: int32 = b - c + i
            return d

        k_foo = self.helper(foo)
        for x, y, z in itertools.product([-50, -1, 0, 6, 32], repeat=3):
            self.assertEqual(foo(x, y, z), k_foo(x, y, z))

    def test_scalar_if3(self):
        def foo(a: int32, b: int32, c: int32) -> int32:
            if a % 2 == 0:
                i: int32 = a % 3
                d: int32 = b + c + i
            else:
                j: int32 = a % 4
                d: int32 = b - c + j
            return d

        k_foo = self.helper(foo)
        for x, y, z in itertools.product([-50, -1, 0, 6, 32], repeat=3):
            self.assertEqual(foo(x, y, z), k_foo(x, y, z))



    def test_scalar_if4(self):
        def foo(a: int32, b: int32, c: int32) -> float64:
            if a % 2 == 0:
                i: int32 = a % 3
                d: float64 = b + c / i
            else:
                i: float64 = a / 4
                d: float64 = b - c / i
            return d

        k_foo = self.helper(foo)
        for x, y, z in itertools.product([-50, -1, 0, 6, 32], repeat=3):
            self.assertEqual(foo(x, y, z), k_foo(x, y, z))
"""

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
