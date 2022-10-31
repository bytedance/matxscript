# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import unittest
from typing import List
from typing import Any
import numpy as np
import matx


class TestNDArrayFusion(unittest.TestCase):

    def test_fusion_getitem(self):

        def fusion_getitem(init_values: List) -> int:
            s = 0
            a = matx.NDArray(init_values, [2, 5], "int32")
            for i in range(2):
                for j in range(5):
                    s += a[i][j]
            return s

        init = [i for i in range(10)]
        exp_ret = sum(init)
        tx_ret = matx.script(fusion_getitem)(init)
        self.assertEqual(tx_ret, exp_ret)

    def test_fusion_setitem(self):

        def fusion_setitem() -> matx.NDArray:
            a = matx.NDArray([], [2, 5], "int32")
            for i in range(2):
                for j in range(5):
                    a[i][j] = i + j
            return a

        tx_ret = matx.script(fusion_setitem)().asnumpy()
        for i in range(2):
            for j in range(5):
                self.assertEqual(tx_ret[i][j], i + j)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
