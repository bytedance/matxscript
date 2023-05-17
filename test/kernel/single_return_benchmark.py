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
import time
from matx.kernel.kernel_parser import KernelParser
from matx.kernel.compile_linalg import compile_linalg
from matx.kernel.typing import int32, int64, float32

WARM_UP = 100
ITERATION = 30000


class TestSingleReturnParser(unittest.TestCase):

    def test_multiple_bin_op_with_scalar_and_const(self):
        M = sympy.Symbol('M', positive=True)
        N = sympy.Symbol('N', positive=True)

        def foo(a: int32[M, N], b: int64[M, N], c: float32) -> float32[M, N]:
            return ((a + b) * c) + 1 + a

        p = KernelParser(foo)
        p.parse()
        
        a = np.arange(300*400, dtype=np.int32).reshape(300, 400)
        b = np.arange(300*400, dtype=np.int64).reshape(300, 400)
        c = np.float32(3)
        rt = np.zeros(a.shape, dtype=np.float32)
        f = compile_linalg(p)
        c_args, _ = f.to_c_args(a, b, c, rt=rt)
        print('-'*100)
        print(*c_args)

        # warmup
        for _ in range(WARM_UP):
            f.raw_call(*c_args)
            
        linalg_start_time = time.time()
        for _ in range(ITERATION):
            f.raw_call(*c_args)
        print("--- linalg time: %s seconds ---" % (time.time() - linalg_start_time))
        
        for _ in range(WARM_UP):
            _ = foo(a, b, c)
        
        numpy_start_time = time.time()
        for _ in range(ITERATION):
            _ = foo(a, b, c)
        print("--- numpy time: %s seconds ---" % (time.time() - numpy_start_time))   
        