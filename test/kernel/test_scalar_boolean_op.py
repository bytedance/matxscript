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


class TestMLIRBooleanOp(unittest.TestCase):
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

    def test_and(self):
        def foo(a: boolean, b: boolean) -> boolean:
            return a and b

        foo = self.helper(foo)
        self.assertFalse(foo(0, 0))
        self.assertFalse(foo(0, 1))
        self.assertFalse(foo(1, 0))
        self.assertTrue(foo(1, 1))
        self.assertFalse(foo(False, False))
        self.assertFalse(foo(False, True))
        self.assertFalse(foo(True, False))
        self.assertTrue(foo(True, True))

    def test_or(self):
        def foo(a: boolean, b: boolean) -> boolean:
            return a or b

        foo = self.helper(foo)
        self.assertFalse(foo(0, 0))
        self.assertTrue(foo(0, 1))
        self.assertTrue(foo(1, 0))
        self.assertTrue(foo(1, 1))
        self.assertFalse(foo(False, False))
        self.assertTrue(foo(False, True))
        self.assertTrue(foo(True, False))
        self.assertTrue(foo(True, True))

    def test_not(self):
        def foo(a: boolean) -> boolean:
            return not a

        foo = self.helper(foo)
        self.assertFalse(foo(True))
        self.assertTrue(foo(False))

    def test_mixed(self):
        def foo(a: boolean, b: boolean, c: boolean) -> boolean:
            return not a


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
