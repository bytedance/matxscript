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
import uuid
import unittest
import matx
from matx import pipeline
from typing import Tuple


@matx.script
class MyOutputOneRes1:

    def __init__(self, bias: int) -> None:
        self.bias: int = bias

    def __call__(self, a: int) -> int:
        return a + self.bias


@matx.script
class MyOutputOneRes2:

    def __init__(self, bias: int) -> None:
        self.bias: int = bias

    def __call__(self, a: int) -> int:
        return a * self.bias


@matx.script
class MyOutputTwoRes:

    def __init__(self, bias: int) -> None:
        self.bias: int = bias

    def __call__(self, a: int) -> Tuple[int, int]:
        return a + self.bias, a * self.bias


class TestSymbolFree(unittest.TestCase):
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    rand = uuid.uuid4().int & 0xFFFFFFFF

    def setUp(self) -> None:

        self.tmp_path = self.cur_path + "/../tempdir/"
        self.data_path = self.cur_path + "/../data/"
        self.work_path = self.tmp_path + "TestSymbolFree_%d/" % self.rand
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)

    def test_single_free(self):
        test_op1 = MyOutputOneRes1(bias=2)
        test_op2 = MyOutputOneRes2(bias=2)

        def workflow(a):
            r = test_op1(a)
            r = test_op2(a)
            r = test_op1(a)
            r = test_op2(a)
            return r

        input_a = 2
        py_res = workflow(input_a)
        jit_mod = pipeline.Trace(workflow, input_a)
        tx_res = jit_mod.run({"a": input_a})
        self.assertEqual(py_res, tx_res)
        save_path = self.work_path + "test_single_free"
        jit_mod.save(save_path)
        jit_mod = matx.pipeline.Load(save_path, -1)
        tx_res = jit_mod.run({"a": input_a})
        self.assertEqual(py_res, tx_res)

    def test_multi_free(self):
        test_op1 = MyOutputOneRes1(bias=2)
        test_op2 = MyOutputOneRes2(bias=2)
        test_op3 = MyOutputTwoRes(bias=2)

        def workflow1(a):
            # r1 should not be free
            r1, r2 = test_op3(a)
            r = test_op1(r2)
            r = test_op2(r2)
            return r

        def workflow2(a):
            # r2 should not be free
            r1, r2 = test_op3(a)
            r = test_op1(r1)
            r = test_op2(r1)
            return r

        def workflow3(a):
            # r1, r2 should not be free
            r1, r2 = test_op3(a)
            r = test_op1(r1)
            r = test_op2(r2)
            return r

        def workflow4(a):
            # r1, r2 should be free
            r1, r2 = test_op3(a)
            r = test_op1(a)
            r = test_op2(a)
            return r

        for workflow in [workflow1, workflow2, workflow3, workflow4]:
            input_a = 2
            py_res = workflow(input_a)
            jit_mod = pipeline.Trace(workflow, input_a)
            tx_res = jit_mod.run({"a": input_a})
            self.assertEqual(py_res, tx_res)
            save_path = self.work_path + "test_multi_free_" + workflow.__name__
            jit_mod.save(save_path)
            jit_mod = matx.pipeline.Load(save_path, -1)
            tx_res = jit_mod.run({"a": input_a})
            self.assertEqual(py_res, tx_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
