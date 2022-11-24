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


class TestCudaCastOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        images = np.array([image, image, image])
        image_float = image.astype("float32")
        images_float = images.astype("float32")

        self.alpha = 1.0 / 255
        self.beta = 0.0
        self.origin_res = (image_float + self.beta) * self.alpha
        self.origin_res_batch = (images_float + self.beta) * self.alpha
        self.image_nd = matx.array.from_numpy(image, "gpu:0")
        self.images_nd = matx.array.from_numpy(images, "gpu:0")
        self.image_nd_cpu = matx.array.from_numpy(image)
        self.images_nd_cpu = matx.array.from_numpy(images)
        self.dtype = "float32"
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def test_gpu_cast_op(self):
        cast_op = byted_vision.CastOp(self.device)

        op_ret = cast_op(self.image_nd, self.dtype, self.alpha, self.beta)
        self._helper(op_ret, self.origin_res)

        op_ret_batch = cast_op(
            self.images_nd, self.dtype, self.alpha, self.beta)
        self._helper(op_ret_batch, self.origin_res_batch)

    def test_scripted_gpu_cast_op(self):
        script_cast_op = matx.script(byted_vision.CastOp)(self.device)

        script_ret = script_cast_op(
            self.image_nd, self.dtype, self.alpha, self.beta)
        self._helper(script_ret, self.origin_res)

        script_ret_batch = script_cast_op(
            self.images_nd, self.dtype, self.alpha, self.beta)
        self._helper(script_ret_batch, self.origin_res_batch)

    def _gpu_cast_op_cpu_input_sync(self, cast_op):
        op_ret = cast_op(self.image_nd_cpu, self.dtype,
                         self.alpha, self.beta, byted_vision.SYNC_CPU)
        self.assertEqual(op_ret.device(), "cpu")
        self._helper(op_ret, self.origin_res)

        op_ret_batch = cast_op(
            self.images_nd_cpu, self.dtype, self.alpha, self.beta, byted_vision.SYNC_CPU)
        self._helper(op_ret_batch, self.origin_res_batch)
        self.assertEqual(op_ret_batch.device(), "cpu")

    def test_gpu_cast_op_cpu_input_sync(self):
        cast_op = byted_vision.CastOp(self.device)
        self._gpu_cast_op_cpu_input_sync(cast_op)

    def test_gpu_cast_op_cpu_input_sync_script(self):
        cast_op = matx.script(byted_vision.CastOp)(self.device)
        self._gpu_cast_op_cpu_input_sync(cast_op)

    def _helper(self, ret, target):
        np.testing.assert_almost_equal(ret.asnumpy(), target)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
