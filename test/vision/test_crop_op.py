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
from matx.vision import CropOp, CenterCropOp

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestCenterCropOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')
        # [360, 640, 3]
        image = cv2.imread(image_file)
        origin_height, origin_width, _ = image.shape
        self.batch_size = 8
        self.image_nds = [matx.array.from_numpy(image)
                          for _ in range(self.batch_size)]
        images = [image] * self.batch_size
        self.device = matx.Device("cpu")
        crop_edge_x = 100
        crop_edge_y = 100

        self.x = [10, 20, 30, 15, 25, 35, 14, 24]
        self.y = [50, 35, 20, 15, 22, 44, 32, 17]
        self.width = [origin_width - crop_edge_x] * self.batch_size
        self.height = [origin_height - crop_edge_y] * self.batch_size

        tmp_idx_top = crop_edge_y // 2
        tmp_idx_left = crop_edge_x // 2

        self.center_crop_res = [images[i][tmp_idx_top:tmp_idx_top +
                                          self.height[i], tmp_idx_left:tmp_idx_left +
                                          self.width[i]] for i in range(self.batch_size)]
        self.custom_crop_res = [images[i][self.y[i]:self.y[i] + self.height[i],
                                          self.x[i]:self.x[i] + self.width[i]] for i in range(self.batch_size)]

        return super().setUp()

    def test_center_crop_op(self):
        center_crop_op = CenterCropOp(
            device=self.device, sizes=(self.height[0], self.width[0]))
        op_ret = center_crop_op(self.image_nds)
        self._helper(op_ret, self.center_crop_res)

    def test_script_center_crop_op(self):
        center_crop_op = matx.script(CenterCropOp)(device=self.device, sizes=(self.height[0], self.width[0]))
        op_ret = center_crop_op(self.image_nds)
        self._helper(op_ret, self.center_crop_res)

    def test_crop_op(self):
        crop_op = CropOp(device=self.device)
        op_ret = crop_op(self.image_nds, self.x, self.y,
                         self.width, self.height)
        self._helper(op_ret, self.custom_crop_res)

    def test_script_crop_op(self):
        crop_op = matx.script(CropOp)(device=self.device)
        op_ret = crop_op(self.image_nds, self.x, self.y,
                         self.width, self.height)
        self._helper(op_ret, self.custom_crop_res)

    def _helper(self, ret, target):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), target[i])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
