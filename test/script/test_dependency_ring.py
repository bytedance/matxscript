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
import matx
from typing import Any, List


class TestDAGRing(unittest.TestCase):

    def test_ring(self):
        def func_a(x: int) -> int:
            if x > 100:
                return x
            return func_c(x + 1)

        def func_b(x: int) -> int:
            if x > 100:
                return x
            return func_a(x + 1)

        def func_c(x: int) -> int:
            if x > 100:
                return x
            return func_b(x + 1)

        py_ret = func_c(1)
        tx_ret = matx.script(func_c)(1)
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
