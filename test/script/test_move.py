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

from typing import List, Dict, Set, Tuple
import unittest
import matx

checker = matx.get_global_func("ir.AssignOptimizer_Match")


class TestAutoMove(unittest.TestCase):
    def test_func_match(self):
        def test_normal_move(a: List[str], b: str, c: int) -> List[str]:
            a1 = a
            b1 = b
            c1 = c
            d = "hello world".split(b)
            return d

        func = matx.ir_module(test_normal_move)["test_normal_move"]
        self.assertTrue(checker(func))

        def test_use_after_move(a: List[str], b: str, c: int) -> List[str]:
            a1 = a
            b1 = b
            c1 = c
            d = a[0].split(b)
            return d

        func = matx.ir_module(test_use_after_move)["test_use_after_move"]
        self.assertTrue(checker(func))

        res = matx.script(test_use_after_move)(["hello world"], " ", 0)
        self.assertEqual(res, ["hello", "world"])

    def test_filter(self):

        def test_func_with_if(a: List[str], b: str, c: int) -> List[str]:
            a1 = a
            b1 = b
            c1 = c
            d = a[0].split(b)
            if False:
                print(d)
            return d

        func = matx.ir_module(test_func_with_if)["test_func_with_if"]
        self.assertFalse(checker(func))

        def test_func_with_for(a: List[str], b: str, c: int) -> List[str]:
            a1 = a
            b1 = b
            c1 = c
            d = a[0].split(b)
            for i in range(1):
                print(d)
            return d

        func = matx.ir_module(test_func_with_for)["test_func_with_for"]
        self.assertFalse(checker(func))

        def test_func_with_while(a: List[str], b: str, c: int) -> List[str]:
            a1 = a
            b1 = b
            c1 = c
            d = a[0].split(b)
            while False:
                print(d)
            return d

        func = matx.ir_module(test_func_with_while)["test_func_with_while"]
        self.assertFalse(checker(func))

        def test_func_with_auto_for(a: List[str], b: str, c: int) -> List[str]:
            a1 = a
            b1 = b
            c1 = c
            d = a[0].split(b)
            for x in ["hello"]:
                print(x)
            return d

        func = matx.ir_module(test_func_with_auto_for)["test_func_with_auto_for"]
        self.assertFalse(checker(func))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
