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
from typing import List, Any, Dict

dev = matx.Device("cpu")

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestTraceFreeVars(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"
        self.work_path = self.tmp_path + "TestTraceCaptureOpKernel_%d/" % uuid.uuid4().int
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)

    def test_capture_prebuilt_op(self):
        def my_capture_native_op() -> Any:
            return dev()

        compiled_op = matx.script(my_capture_native_op)
        py_ret = my_capture_native_op()
        tx_ret = compiled_op()
        self.assertEqual(py_ret, tx_ret)

        def workflow():
            return compiled_op()

        jit_mod = matx.trace(workflow)
        sess_ret = jit_mod.run({})
        self.assertEqual(sess_ret, py_ret)

        save_path = self.work_path + "test_capture_prebuilt_op"
        jit_mod.save(save_path)

        jit_mod_2 = matx.load(save_path, "cpu")
        sess_ret = jit_mod_2.run({})
        print(sess_ret)

    def test_capture_user_op(self):
        class MyResource:
            def __init__(self):
                self.x: int = 1

            def __call__(self) -> int:
                return self.x

        css = matx.script(MyResource)()

        def my_capture_user_op() -> Any:
            return css()

        compiled_op = matx.script(my_capture_user_op)
        py_ret = my_capture_user_op()
        tx_ret = compiled_op()
        self.assertEqual(py_ret, tx_ret)

        def workflow():
            return compiled_op()

        jit_mod = matx.trace(workflow)
        sess_ret = jit_mod.run({})
        self.assertEqual(sess_ret, py_ret)

        save_path = self.work_path + "test_capture_user_op"
        jit_mod.save(save_path)

        jit_mod_2 = matx.load(save_path, "cpu")
        sess_ret = jit_mod_2.run({})
        self.assertEqual(sess_ret, py_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
