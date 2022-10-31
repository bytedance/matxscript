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
from typing import Any


class TestIfThenElse(unittest.TestCase):

    def test_prim_ifexpr(self):
        def return_if_expr(x: int) -> int:
            return x + 1 if x > 3 else x - 1

        py_ret = return_if_expr(1)
        tx_ret = matx.script(return_if_expr)(1)
        self.assertEqual(py_ret, tx_ret)

        def assign_if_expr(x: int) -> int:
            y = x + 1 if x > 3 else x - 1
            return y

        py_ret = assign_if_expr(1)
        tx_ret = matx.script(assign_if_expr)(1)
        self.assertEqual(py_ret, tx_ret)

    def test_hlo_ifexpr(self):
        def return_if_expr(x: Any) -> Any:
            return "hello" if x > 3 else x - 1

        py_ret = return_if_expr(1)
        tx_ret = matx.script(return_if_expr)(1)
        self.assertEqual(py_ret, tx_ret)

        def assign_if_expr(x: Any) -> Any:
            y = "hello" if x > 3 else x - 1
            return y

        py_ret = assign_if_expr(1)
        tx_ret = matx.script(assign_if_expr)(1)
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
