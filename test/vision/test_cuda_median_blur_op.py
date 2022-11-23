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


class TestCudaMedianBlurOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        images = [image, image, image]
        self.batch_size = len(images)
        self.ksizes = [(3, 3), (5, 5), (7, 7)]

        self.origin_res = [
            cv2.medianBlur(
                images[i],
                ksize=self.ksizes[i][0],
            ) for i in range(
                self.batch_size)]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def test_cuda_median_blur_op(self):
        op = byted_vision.MedianBlurOp(self.device)
        op_ret = op(self.image_nd, self.ksizes)
        self._helper(op_ret)

    def test_scripted_cuda_median_blur_op(self):
        script_op = matx.script(byted_vision.MedianBlurOp)(self.device)
        script_ret = script_op(self.image_nd, self.ksizes)
        self._helper(script_ret)

    def _cuda_median_blur_cpu_input_sync(self, op):
        op_ret = op(self.image_nd_cpu, self.ksizes, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret)

    def test_cuda_median_blur_cpu_input_sync(self):
        op = byted_vision.MedianBlurOp(self.device)
        self._cuda_median_blur_cpu_input_sync(op)

    def test_cuda_median_blur_cpu_input_sync_scripted(self):
        op = matx.script(byted_vision.MedianBlurOp)(self.device)
        self._cuda_median_blur_cpu_input_sync(op)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), self.origin_res[i])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
