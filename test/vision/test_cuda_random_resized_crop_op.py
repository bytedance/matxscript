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


class TestCudaBatchRandomResizedCropOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        self.batch_size = 3
        self.scale = [0.8, 1.0]
        self.ratio = [3.0 / 4.0, 4.0 / 3.0]

        image = cv2.imread(image_file)
        images = [image] * self.batch_size
        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]

        scale_percent = [0.5]
        self.size = (int(image.shape[0] * scale_percent[0]),
                     int(image.shape[1] * scale_percent[0]))

        self.interp = byted_vision.INTER_LINEAR
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def test_cuda_random_resized_crop_op(self):
        random_resized_crop_op = byted_vision.RandomResizedCropOp(
            device=self.device, size=self.size, scale=self.scale, ratio=self.ratio, interp=self.interp)
        op_ret = random_resized_crop_op(self.image_nd)
        self._helper(op_ret)

        cpu_ret = random_resized_crop_op(
            self.image_nd_cpu, byted_vision.SYNC_CPU)
        for nd in cpu_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(cpu_ret)

    def test_scripted_cuda_resize_op(self):
        script_resize_op = matx.script(
            byted_vision.RandomResizedCropOp)(
            device=self.device,
            size=self.size,
            scale=self.scale,
            ratio=self.ratio,
            interp=self.interp)
        script_ret = script_resize_op(self.image_nd)
        self._helper(script_ret)

        cpu_ret = script_resize_op(self.image_nd_cpu, byted_vision.SYNC_CPU)
        for nd in cpu_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(cpu_ret)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(ret[i].asnumpy().shape, [
                self.size[0], self.size[1], 3])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
