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


class TestAutoContrastOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.image_nd = matx.array.from_numpy(image, "gpu:0")
        self.device = matx.Device("gpu:0")
        self.origin_res = self.np_auto_contrast(image)

        return super().setUp()

    def np_auto_contrast(self, img):
        hi = np.max(img, (0, 1), keepdims=True)
        lo = np.min(img, (0, 1), keepdims=True)
        scale = 255.0 / np.maximum(hi - lo, 1)
        img = (img - lo) * scale
        return np.clip(img, 0, 255).astype(np.uint8)

    def test_auto_contrast_op(self):
        auto_contrast_op = byted_vision.AutoContrastOp(self.device)
        op_ret = auto_contrast_op([self.image_nd])
        self._helper(op_ret[0])

    def test_scripted_auto_contrast_op(self):
        script_auto_contrast_op = matx.script(byted_vision.AutoContrastOp)(self.device)
        script_ret = script_auto_contrast_op([self.image_nd])
        self._helper(script_ret[0])

    def _cuda_auto_contrast_sync_cpu(self, op):
        op_ret = op([self.image_nd], byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret[0])

    def test_auto_contrast_sync_cpu(self):
        op = byted_vision.AutoContrastOp(self.device)
        self._cuda_auto_contrast_sync_cpu(op)

    def test_auto_contrast_sync_cpu_scripted(self):
        op = matx.script(byted_vision.AutoContrastOp)(self.device)
        self._cuda_auto_contrast_sync_cpu(op)

    def _helper(self, ret):
        res = ret.asnumpy()
        diff = np.abs(res - self.origin_res)
        diff_num = np.sum(diff > 1)
        h, w, c = self.origin_res.shape
        diff_ratio = diff_num / (h * w * c)
        assert diff_ratio < 0.01


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
