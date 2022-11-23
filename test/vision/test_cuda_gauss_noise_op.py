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
import random

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestCudaGaussNoiseOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        images = [image, image, image]
        self.batch_size = len(images)
        self.mu = [0, 0, 0]
        self.sigma = [1, 0.1, 0.01]

        self.origin_res = images

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def test_cuda_gauss_noise_op(self):
        op = byted_vision.GaussNoiseOp(self.device, self.batch_size)
        op_ret = op(self.image_nd, self.mu, self.sigma)
        self._helper(op_ret)

    def test_scripted_cuda_gauss_noise_op(self):
        script_op = matx.script(byted_vision.GaussNoiseOp)(self.device, self.batch_size)
        script_ret = script_op(self.image_nd, self.mu, self.sigma)
        self._helper(script_ret)

    def _cuda_gauss_noise_cpu_input_sync(self, op):
        op_ret = op(self.image_nd_cpu, self.mu, self.sigma, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret)

    def test_cuda_gauss_noise_cpu_input_sync(self):
        op = byted_vision.GaussNoiseOp(self.device, self.batch_size)
        self._cuda_gauss_noise_cpu_input_sync(op)

    def test_cuda_gauss_noise_cpu_input_sync_scripted(self):
        op = matx.script(byted_vision.GaussNoiseOp)(self.device, self.batch_size)
        self._cuda_gauss_noise_cpu_input_sync(op)

    def _helper(self, ret):
        for i in range(self.batch_size):
            # since the noise is random, we cannot easily test the image value here
            # we just test the output shape for now
            assert ret[i].asnumpy().shape == self.origin_res[i].shape


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
