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
from typing import Any

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class MyInfo:

    def __init__(self, a1: int, b1: int, c1: int) -> None:
        self.a: int = a1
        self.b: int = b1
        self.c: int = c1

    def get_result(self) -> int:
        return self.a + self.b + self.c


class MyOp1:

    def __init__(self) -> None:
        self.op1_attr: int = 10

    def __call__(self, info: MyInfo) -> int:
        return info.get_result() + 10


class MyOp2:

    def __init__(self) -> None:
        self.op2_attr: int = 10

    def __call__(self, info: MyInfo) -> int:
        # info2 = MyInfo(3, 4, 5)
        # return info.get_result() + info2.get_result()
        return info.get_result() + 5


class MyOp3:

    def __init__(self) -> None:
        self.op3_attr: int = 10

    def __call__(self, info: MyInfo) -> Any:
        return info


def my_global_func() -> int:
    return 10


class MyOpSum:

    def __init__(self) -> None:
        self.op_sum_attr: int = 10

    def __call__(self, info1: Any, info2: Any) -> int:
        info3 = MyInfo(3, 4, 5)
        x = my_global_func
        return x() + info1.get_result() + info1.a + info2.get_result() + info3.get_result()


class TestClassAsPipelineArgs(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir"
        self.module_path = self.tmp_path + os.sep + "TestClassAsPipelineArgs_%d" % uuid.uuid4().int

    def test_op_process_class(self):
        def raw_pipeline(ud1: MyInfo):
            op1 = MyOp1()
            op2 = MyOp2()
            res1 = op1(ud1)
            res2 = op2(ud1)
            return res1, res2

        def trace_pipeline(ud1: MyInfo):
            op1 = matx.script(MyOp1)()
            op2 = matx.script(MyOp2)()
            res1 = op1(ud1)
            res2 = op2(ud1)
            return res1, res2

        raw_ud = MyInfo(1, 1, 1)
        ans1, ans2 = raw_pipeline(raw_ud)
        ud_creator = matx.script(MyInfo)
        compiled_ud = ud_creator(1, 1, 1)
        res1, res2 = trace_pipeline(compiled_ud)
        print(res1, res2)
        self.assertEqual((res1, res2), (ans1, ans2))

        compiled_ud = ud_creator(1, 1, 1)
        jit_module = matx.pipeline.Trace(trace_pipeline, ud1=compiled_ud)
        compiled_ud = ud_creator(1, 1, 1)
        res1, res2 = jit_module.run(feed_dict={"ud1": compiled_ud})
        print(res1, res2)
        self.assertEqual((res1, res2), (ans1, ans2))
        jit_module.save(self.module_path)

        jit_module = matx.pipeline.Load(self.module_path, -1)
        compiled_ud = ud_creator(1, 1, 1)
        res1, res2 = jit_module.run(feed_dict={"ud1": compiled_ud})
        print(res1, res2)
        self.assertEqual((res1, res2), (ans1, ans2))

    def test_op_process_generic_class(self):
        def raw_pipeline(ud1: MyInfo):
            op1 = MyOp3()
            op2 = MyOp3()
            op_sum = MyOpSum()
            res1 = op1(ud1)
            res2 = op2(ud1)
            res = op_sum(res1, res2)
            return res

        def trace_pipeline(ud1: MyInfo):
            op_sum = matx.script(MyOpSum)()
            op1 = matx.script(MyOp3)()
            op2 = matx.script(MyOp3)()
            res1 = op1(ud1)
            res2 = op2(ud1)
            res = op_sum(res1, res2)
            return res

        raw_ud = MyInfo(1, 1, 1)
        ans = raw_pipeline(raw_ud)
        ud_creator = matx.script(MyInfo)
        compiled_ud = ud_creator(1, 1, 1)
        res = trace_pipeline(compiled_ud)
        print(res)
        self.assertEqual(res, ans)

        compiled_ud = ud_creator(1, 1, 1)
        jit_module = matx.pipeline.Trace(trace_pipeline, ud1=compiled_ud)
        compiled_ud = ud_creator(1, 1, 1)
        res = jit_module.run(feed_dict={"ud1": compiled_ud})
        print(res)
        self.assertEqual(res, ans)
        jit_module.save(self.module_path)

        jit_module = matx.pipeline.Load(self.module_path, -1)
        compiled_ud = ud_creator(1, 1, 1)
        res = jit_module.run(feed_dict={"ud1": compiled_ud})
        print(res)
        self.assertEqual(res, ans)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
