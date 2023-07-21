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

import matx.kernel
from matx.kernel.typing import int32


class TestKernelTemplateFunc(unittest.TestCase):

    def test_simple_template(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        @matx.kernel.template
        def foo(a, b):
            return a + b

        @matx.kernel.func
        def k_boo(a: int32[M, N]) -> int32[M, N]:
            return foo(a, a)

        def boo(a: int32[M, N]) -> int32[M, N]:
            return foo(a, a)

        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        np.testing.assert_equal(k_boo(a), boo(a))

    def test_simple_template2(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a, b):
            c = a + 1
            return c + b

        @matx.kernel.template
        def foo(a, b):
            c = a + 1
            return c + b

        @matx.kernel.func
        def k_boo(a: int32[M, N]) -> int32[M, N]:
            return foo(a, a)

        def boo(a: int32[M, N]) -> int32[M, N]:
            return foo(a, a)

        c = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        np.testing.assert_equal(k_boo(c), boo(c))
