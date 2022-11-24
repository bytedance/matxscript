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


class TestResizeOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        self.batch_size = 3

        image = cv2.imread(image_file)
        self.ori_height = image.shape[0]
        self.ori_width = image.shape[1]
        self.ori_ratio = self.ori_width / self.ori_height
        images = [image] * self.batch_size
        self.image_nd = [matx.array.from_numpy(i) for i in images]

        self.interp = byted_vision.INTER_LINEAR

        self.device = matx.Device("cpu")

        return super().setUp()

    def test_resize_op_default_mode_unique_size(self):
        mode = byted_vision.RESIZE_DEFAULT
        size = (224, 224)
        desired_sizes = [(224, 224), (224, 224), (224, 224)]

        resize_op = byted_vision.ResizeOp(self.device, size=size)
        op_ret = resize_op(self.image_nd)
        self._helper(op_ret, desired_sizes)

        script_resize_op = matx.script(byted_vision.ResizeOp)(self.device, size=size)
        script_ret = script_resize_op(self.image_nd)
        self._helper(script_ret, desired_sizes)

    def test_resize_op_default_mode_various_size(self):
        mode = byted_vision.RESIZE_DEFAULT
        sizes = [(100, 100), (100, 150), (200, 100)]
        desired_sizes = [(100, 100), (100, 150), (200, 100)]

        resize_op = byted_vision.ResizeOp(self.device)
        op_ret = resize_op(self.image_nd, sizes)
        self._helper(op_ret, desired_sizes)

        script_resize_op = matx.script(byted_vision.ResizeOp)(self.device)
        script_ret = script_resize_op(self.image_nd, sizes)
        self._helper(script_ret, desired_sizes)

    def test_resize_op_not_smaller_mode_various_size(self):
        mode = byted_vision.RESIZE_NOT_SMALLER
        max_size = 180
        sizes = [(100, 100), (100, 150), (200, 100)]
        desired_sizes = [(100, int(100 * self.ori_ratio)),
                         (100, int(100 * self.ori_ratio)),
                         (int(180 / self.ori_ratio), 180)]
        resize_op = byted_vision.ResizeOp(self.device, max_size=max_size, mode=mode)
        op_ret = resize_op(self.image_nd, sizes)
        self._helper(op_ret, desired_sizes)

        script_resize_op = matx.script(
            byted_vision.ResizeOp)(
            self.device,
            max_size=max_size,
            mode=mode)
        script_ret = script_resize_op(self.image_nd, sizes)
        self._helper(script_ret, desired_sizes)

    def test_resize_op_not_larger_mode_various_size(self):
        mode = byted_vision.RESIZE_NOT_LARGER
        sizes = [(100, 100), (100, 150), (200, 100)]
        desired_sizes = [(int(100 / self.ori_ratio), 100),
                         (int(150 / self.ori_ratio), 150),
                         (int(100 / self.ori_ratio), 100)]
        resize_op = byted_vision.ResizeOp(self.device, mode=mode)
        op_ret = resize_op(self.image_nd, sizes)
        self._helper(op_ret, desired_sizes)

        script_resize_op = matx.script(byted_vision.ResizeOp)(self.device, mode=mode)
        script_ret = script_resize_op(self.image_nd, sizes)
        self._helper(script_ret, desired_sizes)

    def _helper(self, ret, sizes):
        for i in range(self.batch_size):
            res = ret[i].asnumpy()
            assert res.shape[0] == sizes[i][0]
            assert res.shape[1] == sizes[i][1]


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
