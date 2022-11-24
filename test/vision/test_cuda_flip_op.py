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


class TestCudaFlipOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        images = [image, image, image]
        self.batch_size = len(images)
        self.flip_code_list = [
            byted_vision.DIAGONAL_FLIP,
            byted_vision.VERTICAL_FLIP,
            byted_vision.HORIZONTAL_FLIP
        ]
        self.origin_res_1 = [cv2.flip(images[i], self.flip_code_list[i])
                             for i in range(self.batch_size)]
        self.flip_code = byted_vision.HORIZONTAL_FLIP
        self.origin_res_2 = [cv2.flip(images[i], self.flip_code)
                             for i in range(self.batch_size)]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.device = matx.Device("gpu:0")
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]

        return super().setUp()

    def test_cuda_flip_op(self):
        batch_flip_op = byted_vision.FlipOp(self.device, self.flip_code)
        op_ret = batch_flip_op(self.image_nd, self.flip_code_list)
        self._helper(op_ret, self.origin_res_1)

        op_ret = batch_flip_op(self.image_nd)
        self._helper(op_ret, self.origin_res_2)

    def test_scripted_cuda_flip_op(self):
        batch_flip_op = matx.script(
            byted_vision.FlipOp)(
            self.device,
            self.flip_code)
        op_ret = batch_flip_op(self.image_nd, self.flip_code_list)
        self._helper(op_ret, self.origin_res_1)

        op_ret = batch_flip_op(self.image_nd)
        self._helper(op_ret, self.origin_res_2)

    def test_cuda_flip_op_cpu_sync(self):
        batch_flip_op = byted_vision.FlipOp(
            self.device,
            self.flip_code)
        op_ret = batch_flip_op(
            self.image_nd, self.flip_code_list, byted_vision.SYNC_CPU)
        self.assertEqual(op_ret[0].device(), "cpu")
        self._helper(op_ret, self.origin_res_1)

        op_ret = batch_flip_op(self.image_nd, [], byted_vision.SYNC_CPU)
        self.assertEqual(op_ret[0].device(), "cpu")
        self._helper(op_ret, self.origin_res_2)

    def test_cuda_flip_op_cpu_input(self):
        batch_flip_op = byted_vision.FlipOp(
            self.device,
            self.flip_code)
        op_ret = batch_flip_op(self.image_nd_cpu, self.flip_code_list)
        self._helper(op_ret, self.origin_res_1)

        op_ret = batch_flip_op(self.image_nd_cpu)
        self._helper(op_ret, self.origin_res_2)

    def _helper(self, ret, origin_res):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), origin_res[i])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
