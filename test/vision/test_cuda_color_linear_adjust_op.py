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


class TestColorLinearAdjustOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        self.image = cv2.imread(image_file)
        self.image_nd = matx.array.from_numpy(self.image, "gpu:0")
        self.image_nd_cpu = matx.array.from_numpy(self.image)
        self.device = matx.Device("gpu:0")
        self.factors = [1.2, 1.1, 0.8]
        self.shifts = [5, -8, 10]
        return super().setUp()

    def uint8clip(self, img):
        return np.clip(img, 0, 255).astype(np.uint8)

    def create_origin_res(self, per_channel):
        if per_channel:
            return self.uint8clip(self.image * self.factors + self.shifts)
        else:
            return self.uint8clip(self.image * self.factors[0] + self.shifts[0])

    def _color_adjust_helper(self, op, per_channel=False):
        if per_channel:
            op_ret = op([self.image_nd], self.factors, self.shifts)
        else:
            op_ret = op([self.image_nd], self.factors[:1], self.shifts[:1])
        origin_res = self.create_origin_res(per_channel)
        self._helper(op_ret[0], origin_res)

    def _color_adjust_cpu_sync_helper(self, op, per_channel=False):
        if per_channel:
            op_ret = op([self.image_nd_cpu], self.factors, self.shifts, byted_vision.SYNC_CPU)
        else:
            op_ret = op([self.image_nd_cpu], self.factors[:1],
                        self.shifts[:1], byted_vision.SYNC_CPU)
        origin_res = self.create_origin_res(per_channel)
        self.assertEqual(op_ret[0].device(), "cpu")
        self._helper(op_ret[0], origin_res)

    def test_color_adjust(self):
        color_adjust = byted_vision.ColorLinearAdjustOp(self.device)
        self._color_adjust_helper(color_adjust)
        self._color_adjust_cpu_sync_helper(color_adjust)
        color_adjust_per = byted_vision.ColorLinearAdjustOp(self.device, per_channel=True)
        self._color_adjust_helper(color_adjust_per, True)
        self._color_adjust_cpu_sync_helper(color_adjust_per, True)
        script_color_adjust = matx.script(byted_vision.ColorLinearAdjustOp)(self.device)
        self._color_adjust_helper(script_color_adjust)
        self._color_adjust_cpu_sync_helper(script_color_adjust)
        script_color_adjust_per = matx.script(
            byted_vision.ColorLinearAdjustOp)(
            self.device, per_channel=True)
        self._color_adjust_helper(script_color_adjust_per, True)
        self._color_adjust_cpu_sync_helper(script_color_adjust_per, True)

    def _helper(self, ret, origin_res):
        res = ret.asnumpy()
        diff = np.sum(np.abs(res - origin_res) > 1)
        assert diff < 10


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
