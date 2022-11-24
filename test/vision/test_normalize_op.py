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
from matx import vision

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestNormalizeOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        images = [image, image, image]
        self.batch_size = len(images)
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.dtype = "float32"
        self.origin_res = [(i - self.mean) / self.std for i in images]

        self.image_nd = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("cpu")

        return super().setUp()

    def test_normalize_op(self):
        norm_op = vision.NormalizeOp(self.device, self.mean, self.std, dtype=self.dtype)
        op_ret = norm_op(self.image_nd)
        self._helper(op_ret)

    def test_scripted_normalize_op(self):
        norm_script_op = matx.script(vision.NormalizeOp)(
            self.device, self.mean, self.std, dtype=self.dtype)
        script_ret = norm_script_op(self.image_nd)
        self._helper(script_ret)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(ret[i].asnumpy(), self.origin_res[i], decimal=5)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
