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

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class MyUserData:

    def __init__(self, a1: int) -> None:
        self.a: int = a1

    def get(self) -> int:
        return self.a


def make_my_user_data(i: int) -> MyUserData:
    return MyUserData(i)


class TestTraceClassMethod(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.module_path = self.tmp_path + "TestTraceClassMethod_%d" % uuid.uuid4().int

    def test_trace(self):
        tx_op = matx.script(make_my_user_data)

        def py_workflow(input_0: int) -> int:
            data = make_my_user_data(input_0)
            return data.get()

        def trace_workflow(input_0: int) -> int:
            data = tx_op(input_0)
            return data.get()

        input_example = 1
        py_ret = py_workflow(input_example)
        jit_module = matx.trace(trace_workflow, input_example)
        ret = jit_module.run(feed_dict={"input_0": input_example})
        self.assertEqual(ret, py_ret)
        jit_module.save(self.module_path)

        jit_module = matx.load(self.module_path, -1)
        ret = jit_module.run(feed_dict={"input_0": input_example})
        self.assertEqual(ret, py_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
