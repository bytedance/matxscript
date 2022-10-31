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

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class TestUnpackContainer(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"
        self.work_path = self.tmp_path + "TestUnpackContainer_%d/" % uuid.uuid4().int
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)

    def test_unpack_list(self):
        def return_3_results() -> Any:
            return [1, 2, 3]

        def py_pipeline():
            a, b, c = return_3_results()
            return a, b, c

        def tx_pipeline():
            a, b, c = matx.script(return_3_results)()
            return a, b, c

        py_ret = py_pipeline()
        mod = matx.trace(tx_pipeline, )
        tx_ret = mod.run({})
        self.assertEqual(py_ret, tx_ret)
        save_path = self.work_path + "/test_unpack_list"
        mod.save(save_path)

        mod = matx.load(save_path, "cpu")
        tx_ret = mod.run({})
        self.assertEqual(py_ret, tx_ret)

    def test_unpack_tuple(self):
        def return_3_results() -> Any:
            return 1, 2, 3

        def py_pipeline():
            a, b, c = return_3_results()
            return a, b, c

        def tx_pipeline():
            a, b, c = matx.script(return_3_results)()
            return a, b, c

        py_ret = py_pipeline()
        mod = matx.trace(tx_pipeline, )
        tx_ret = mod.run({})
        self.assertEqual(py_ret, tx_ret)
        save_path = self.work_path + "/test_unpack_list"
        mod.save(save_path)

        mod = matx.load(save_path, "cpu")
        tx_ret = mod.run({})
        self.assertEqual(py_ret, tx_ret)

    def test_unpack_dynamic(self):
        def make_list(return_num: int) -> Any:
            return ["hi"] * return_num

        def my_loop_fn(n: int) -> Any:
            cons = matx.script(make_list)(n)
            ret = []
            for x in cons:
                ret.append(x)
            return tuple(ret)

        jit_mod = matx.trace(my_loop_fn, 3)
        print("my_loop_fn: Run 3", jit_mod.Run({"n": 3}))
        with self.assertRaises(ValueError):
            print("my_loop_fn: Run 4", jit_mod.Run({"n": 4}))
        with self.assertRaises(ValueError):
            print("my_loop_fn: Run 2", jit_mod.Run({"n": 2}))

        def my_unpack_fn(n: int) -> Any:
            a, d, h = matx.script(make_list)(n)
            return a, d, h

        jit_mod = matx.trace(my_unpack_fn, 3)
        print("my_unpack_fn: Run 3", jit_mod.Run({"n": 3}))
        with self.assertRaises(ValueError):
            print("my_unpack_fn: Run 4", jit_mod.Run({"n": 4}))
        with self.assertRaises(ValueError):
            print("my_unpack_fn: Run 2", jit_mod.Run({"n": 2}))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
