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
from matx import pipeline
from typing import Dict
from typing import Tuple
from typing import Any


class TestBasicPipeline(unittest.TestCase):

    def test_one_function_op(self):
        @matx.script
        def test_op(x: int) -> int:
            return x * 10

        def workflow(x):
            return test_op(x)

        jit_mod = pipeline.Trace(workflow, 100)
        ret = jit_mod.run({"x": 100})
        self.assertEqual(ret, 1000)

    def test_simple_class_op(self):
        class LookupTable:

            def __init__(self) -> None:
                self.table: Dict[str, int] = {"a": 1, "b": 2}

            def __call__(self, x: str) -> int:
                return self.table.get(x)

        lookup_op = matx.script(LookupTable)()
        print(lookup_op('a'))

        def workflow(x):
            return lookup_op(x)

        jit_mod = pipeline.Trace(workflow, "a")
        ret = jit_mod.run({"x": "a"})
        self.assertEqual(ret, 1)

    def test_simple_class_with_arg_op(self):
        class LookupTable:

            def __init__(self, table: Any) -> None:
                self.table: Dict[str, int] = table

            def __call__(self, x: str) -> int:
                return self.table.get(x)

        lookup_op = matx.script(LookupTable)(table={"a": 1, "b": 2})
        print(lookup_op('a'))

        def workflow(x):
            return lookup_op(x)

        jit_mod = pipeline.Trace(workflow, "a")
        ret = jit_mod.run({"x": "a"})
        self.assertEqual(ret, 1)

    def test_multi_output(self):
        @matx.script
        class MyMultiOutputOp:

            def __init__(self, bias: int) -> None:
                self.bias: int = bias

            def __call__(self, a: int) -> Tuple[int, int]:
                return a + self.bias, a * self.bias

        @matx.script
        class MySingleOutputOp:

            def __init__(self, bias: int) -> None:
                self.bias1: int = bias

            def __call__(self, b: int) -> int:
                return b * self.bias1

        test_op1 = MyMultiOutputOp(bias=2)
        test_op2 = MySingleOutputOp(bias=4)
        test_op3 = MySingleOutputOp(bias=8)

        def workflow(a):
            ret_add, ret_mul = test_op1(a)
            ret_add2 = test_op2(ret_add)
            ret_mul2 = test_op3(ret_mul)
            return ret_add2, ret_mul2

        jit_mod = pipeline.Trace(workflow, 2)
        ret1, ret2 = jit_mod.run({"a": 2})
        self.assertEqual(ret1, 16)
        self.assertEqual(ret2, 32)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
