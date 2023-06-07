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
from matx.kernel.typing import int32, float32, boolean


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
        # a = 1
        # b = 2
        f = compile_linalg(p)
        # f(a, b)
        # np.testing.assert_equal(rt, foo(a))

    def test_int_eq(self):
        def foo(a: int32, b: int32) -> boolean:
            return a == b

        self.helper(foo)

    def test_int_ne(self):
        def foo(a: int32, b: int32) -> boolean:
            return a != b

        self.helper(foo)

    def test_int_lt(self):
        def foo(a: int32, b: int32) -> boolean:
            return a < b

        self.helper(foo)

    def test_int_le(self):
        def foo(a: int32, b: int32) -> boolean:
            return a <= b

        self.helper(foo)

    def test_int_gt(self):
        def foo(a: int32, b: int32) -> boolean:
            return a > b

        self.helper(foo)

    def test_int_ge(self):
        def foo(a: int32, b: int32) -> boolean:
            return a >= b

        self.helper(foo)


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
        # a = 1
        # b = 2
        f = compile_linalg(p)
        # f(a, b)
        # np.testing.assert_equal(rt, foo(a))

    def test_float_eq(self):
        def foo(a: float32, b: float32) -> boolean:
            return a == b

        self.helper(foo)

    def test_float_ne(self):
        def foo(a: float32, b: float32) -> boolean:
            return a != b

        self.helper(foo)

    def test_float_lt(self):
        def foo(a: float32, b: float32) -> boolean:
            return a < b

        self.helper(foo)

    def test_float_le(self):
        def foo(a: float32, b: float32) -> boolean:
            return a <= b

        self.helper(foo)

    def test_float_gt(self):
        def foo(a: float32, b: float32) -> boolean:
            return a > b

        self.helper(foo)

    def test_float_ge(self):
        def foo(a: float32, b: float32) -> boolean:
            return a >= b

        self.helper(foo)
