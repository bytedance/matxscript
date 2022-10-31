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

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class MyUserData:

    def __init__(self, a1: int, b1: int, c1: int) -> None:
        self.a: int = a1
        self.b: int = b1
        self.c: int = c1

    def forward(self) -> int:
        return self.a + self.b + self.c


class MyOp1:

    def __init__(self, ud1: MyUserData, other1: int) -> None:
        self.ud: MyUserData = ud1
        self.other: int = other1

    def __call__(self, seed: int) -> int:
        return self.ud.forward() + self.other + seed


class MyOp2:

    def __init__(self, ud1: MyUserData, other1: int) -> None:
        self.ud: MyUserData = ud1
        self.other: int = other1

    def __call__(self, seed: int) -> int:
        return self.ud.forward() + self.other + seed + 10


class TestSharedResourceInPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.module_path = self.tmp_path + "TestSharedResourceInPipeline_%d" % uuid.uuid4().int

    def test_pipeline_shared_resource(self):
        def raw_workflow(seed: int):
            ud1 = MyUserData(1, 1, 1)
            op1 = MyOp1(ud1, 2)
            op2 = MyOp2(ud1, 2)
            res1 = op1(seed)
            res2 = op2(seed)
            return res1, res2

        def trace_workflow(seed: int):
            ud1 = matx.script(MyUserData)(1, 1, 1)
            op1 = matx.script(MyOp1)(ud1, 2)
            op2 = matx.script(MyOp2)(ud1, 2)
            res1 = op1(seed)
            res2 = op2(seed)
            return res1, res2

        seed = 10
        ans1, ans2 = raw_workflow(seed)

        jit_module = matx.pipeline.Trace(trace_workflow, seed)
        res1, res2 = jit_module.run(feed_dict={"seed": seed})
        print(res1, res2)
        self.assertEqual((res1, res2), (ans1, ans2))
        jit_module.save(self.module_path)

        jit_module = matx.pipeline.Load(self.module_path, -1)
        res1, res2 = jit_module.run(feed_dict={"seed": seed})
        print(res1, res2)
        self.assertEqual((res1, res2), (ans1, ans2))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
