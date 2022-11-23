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
import cv2
import numpy as np
import matx
from matx import vision as byted_vision

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestStackOp(unittest.TestCase):
    def setUp(self):
        np_data = [np.random.rand(240, 360, 3).astype("float32")
                   for _ in range(5)]
        self.device = matx.Device("gpu:0")
        self.data = [matx.array.from_numpy(i, "gpu:0") for i in np_data]
        self.data_cpu = [matx.array.from_numpy(i) for i in np_data]
        self.origin_res = np.stack(np_data)

    def test_stack_op(self):
        op = byted_vision.StackOp(self.device)
        op_ret = op(self.data, byted_vision.SYNC)
        self._helper(op_ret)

        cpu_ret = op(self.data_cpu, byted_vision.SYNC_CPU)
        self.assertEqual(cpu_ret.device(), "cpu")
        self._helper(cpu_ret)

    def test_scripted_stack_op(self):
        op = matx.script(byted_vision.StackOp)(self.device)
        op_ret = op(self.data, byted_vision.SYNC)
        self._helper(op_ret)

        cpu_ret = op(self.data_cpu, byted_vision.SYNC_CPU)
        self.assertEqual(cpu_ret.device(), "cpu")
        self._helper(cpu_ret)

    def _helper(self, ret):
        np.testing.assert_almost_equal(ret.asnumpy(), self.origin_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
