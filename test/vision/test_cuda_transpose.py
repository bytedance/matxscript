# -*- coding: utf-8 -*-
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


class TestTransposeOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')
        # [360, 640, 3]
        self.batch_size = 4
        image = cv2.imread(image_file)
        batch_image = np.stack([image, image, image, image])
        self.org_image = np.transpose(batch_image, (0, 3, 1, 2))
        self.image_nd = matx.array.from_numpy(batch_image, "gpu:0")
        self.image_nd_cpu = matx.array.from_numpy(batch_image)
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def test_transpose_op(self):
        transpose_op = byted_vision.TransposeOp(device=self.device,
                                                input_layout=byted_vision.NHWC,
                                                output_layout=byted_vision.NCHW)
        op_ret = transpose_op(self.image_nd)
        self._helper(op_ret)

        cpu_ret = transpose_op(self.image_nd_cpu, byted_vision.SYNC_CPU)
        self.assertEqual(cpu_ret.device(), "cpu")
        self._helper(cpu_ret)

    def test_script_transpose_op(self):
        transpose_op = matx.script(byted_vision.TransposeOp)(device=self.device,
                                                             input_layout=byted_vision.NHWC,
                                                             output_layout=byted_vision.NCHW)
        op_ret = transpose_op(self.image_nd)
        self._helper(op_ret)

        cpu_ret = transpose_op(self.image_nd_cpu, byted_vision.SYNC_CPU)
        self.assertEqual(cpu_ret.device(), "cpu")
        self._helper(cpu_ret)

    def _helper(self, ret):
        np.testing.assert_almost_equal(ret.asnumpy(), self.org_image)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
