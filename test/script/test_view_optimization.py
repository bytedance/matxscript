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

import os
import unittest
import matx

from typing import Any, List


class TestIsinstance(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_opt_loop_var(self):
        def make_str(s: str) -> str:
            return "[CLS] " + s

        def opt_loop_var(a: List[Any], cond: bool) -> Any:
            d = []
            for c in a:
                if cond:
                    c = make_str(c)
                else:
                    c = make_str("hi")
                d.append(c)
            return d

        test_case = ["hello", "world"]
        py_ret = opt_loop_var(test_case, True)
        tx_ret = matx.script(opt_loop_var)(test_case, True)
        self.assertSequenceEqual(py_ret, tx_ret)

    def test_reassign_view_var_to_new_var(self):
        def reassign_view_var_to_new_var(s: str) -> str:
            d = s
            d += " [SEP]"
            return "[CLS] " + d

        test_case = "hello world"
        py_ret = reassign_view_var_to_new_var(test_case)
        tx_ret = matx.script(reassign_view_var_to_new_var)(test_case)
        self.assertSequenceEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
