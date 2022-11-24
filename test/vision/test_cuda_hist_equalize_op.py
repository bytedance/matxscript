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


class TestHistEqualizeOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.image_nd = matx.array.from_numpy(image, "gpu:0")
        self.device = matx.Device("gpu:0")
        self.origin_res = self.np_equalize(image)

        return super().setUp()

    def np_equalize(self, img):
        for i in range(img.shape[-1]):
            image_histogram, bins = np.histogram(img[:, :, i], 256, density=True)
            cdf = image_histogram.cumsum()
            cdf = 255 * cdf / cdf[-1]
            image_equalized = np.interp(img[:, :, i].flatten(), bins[:-1], cdf)
            img[:, :, i] = image_equalized.reshape(img.shape[:2])
        return np.clip(img, 0, 255).astype(np.uint8)

    def test_hist_equalize_op(self):
        hist_equalize_op = byted_vision.HistEqualizeOp(self.device)
        op_ret = hist_equalize_op([self.image_nd])
        self._helper(op_ret[0])

    def test_scripted_hist_equalize_op(self):
        script_hist_equalize_op = matx.script(byted_vision.HistEqualizeOp)(self.device)
        script_ret = script_hist_equalize_op([self.image_nd])
        self._helper(script_ret[0])

    def _cuda_hist_equalize_sync_cpu(self, op):
        op_ret = op([self.image_nd], byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret[0])

    def test_hist_equalize_sync_cpu(self):
        op = byted_vision.HistEqualizeOp(self.device)
        self._cuda_hist_equalize_sync_cpu(op)

    def test_hist_equalize_sync_cpu_scripted(self):
        op = matx.script(byted_vision.HistEqualizeOp)(self.device)
        self._cuda_hist_equalize_sync_cpu(op)

    def _helper(self, ret):
        res = ret.asnumpy()
        assert res.shape == self.origin_res.shape
        # np.testing.assert_almost_equal(ret.asnumpy(), self.origin_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
