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
import byted_vision

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestCudaBilateralFilterOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file).astype("uint8")
        images = [image, image, image]
        self.d = [8, 9, 10]
        self.sigma_color = [30.0, 40.0, 50.0]
        self.sigma_space = [30.0, 40.0, 50.0]
        self.batch_size = len(images)
        h, w, _ = image.shape
        self.size = h * w

        self.origin_res = [
            cv2.bilateralFilter(
                images[i],
                self.d[i],
                self.sigma_color[i],
                self.sigma_space[i]) for i in range(
                self.batch_size)]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def test_cuda_bilateral_filter_op(self):
        op = byted_vision.BilateralFilterOp(self.device)
        op_ret = op(self.image_nd, self.d, self.sigma_color, self.sigma_space)
        self._helper(op_ret)

    def test_scripted_cuda_bilateral_filter_op(self):
        script_op = matx.script(byted_vision.BilateralFilterOp)(self.device)
        script_ret = script_op(self.image_nd, self.d,
                               self.sigma_color, self.sigma_space)
        self._helper(script_ret)

    def _cuda_bilateral_filter_cpu_input_sync(self, op):
        op_ret = op(self.image_nd_cpu, self.d, self.sigma_color,
                    self.sigma_space, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret)

    def test_cuda_bilateral_filter_cpu_input_sync(self):
        op = byted_vision.BilateralFilterOp(self.device)
        self._cuda_bilateral_filter_cpu_input_sync(op)

    def test_cuda_bilateral_filter_cpu_input_sync_scripted(self):
        op = matx.script(byted_vision.BilateralFilterOp)(self.device)
        self._cuda_bilateral_filter_cpu_input_sync(op)

    def _helper(self, ret):
        for i in range(self.batch_size):
            cuda_res = ret[i].asnumpy().astype("int")
            cv_res = self.origin_res[i].astype("int")
            diff = np.abs(cuda_res - cv_res)
            self.assertLess(np.count_nonzero(diff == 1), 0.02 * self.size)
            self.assertLess(np.count_nonzero(diff > 1), 1)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
