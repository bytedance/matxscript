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


class TestPosterizeOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.image_nd = matx.array.from_numpy(image, "gpu:0")
        self.device = matx.Device("gpu:0")
        self.bits = 4
        self.origin_res = self.np_posterize(image, self.bits)

        return super().setUp()

    def np_posterize(self, img, bits):
        mask = ~(2 ** (8 - bits) - 1)
        return np.clip(img & mask, 0, 255).astype(np.uint8)

    def test_posterize_op(self):
        posterize_op = byted_vision.PosterizeOp(self.device)
        op_ret = posterize_op([self.image_nd], [self.bits])
        self._helper(op_ret[0])

    def test_scripted_posterize_op(self):
        script_posterize_op = matx.script(byted_vision.PosterizeOp)(self.device)
        script_ret = script_posterize_op([self.image_nd], [self.bits])
        self._helper(script_ret[0])

    def _cuda_posterize_sync_cpu(self, op):
        op_ret = op([self.image_nd], [self.bits], byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret[0])

    def test_posterize_sync_cpu(self):
        op = byted_vision.PosterizeOp(self.device)
        self._cuda_posterize_sync_cpu(op)

    def test_posterize_sync_cpu_scripted(self):
        op = matx.script(byted_vision.PosterizeOp)(self.device)
        self._cuda_posterize_sync_cpu(op)

    def _helper(self, ret):
        np.testing.assert_almost_equal(ret.asnumpy(), self.origin_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
