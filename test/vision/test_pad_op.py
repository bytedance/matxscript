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


class TestPadOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')
        # [360, 640, 3]
        image = cv2.imread(image_file)

        origin_res = cv2.copyMakeBorder(
            image, 10, 10, 20, 20, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self.batch_size = 8
        self.image_nds = [matx.array.from_numpy(image) for _ in range(self.batch_size)]
        self.origin_res = [origin_res for _ in range(self.batch_size)]
        self.device = matx.Device("cpu")

        return super().setUp()

    def test_pad_op(self):
        pad_op = vision.PadOp(device=self.device, size=(
            380, 680), pad_values=(114, 114, 114), pad_type=vision.BORDER_CONSTANT)
        op_ret = pad_op(self.image_nds)
        self._helper(op_ret)

    def test_script_pad_op(self):
        pad_op = matx.script(
            vision.PadOp)(
            device=self.device, size=(
                380, 680), pad_values=(
                114, 114, 114), pad_type=vision.BORDER_CONSTANT)
        op_ret = pad_op(self.image_nds)
        self._helper(op_ret)

    def test_pad_with_border_op(self):
        top_pads = [10] * self.batch_size
        bottom_pads = [10] * self.batch_size
        left_pads = [20] * self.batch_size
        right_pads = [20] * self.batch_size

        pad_op = vision.PadWithBorderOp(device=self.device,
                                              pad_values=(114, 114, 114),
                                              pad_type=vision.BORDER_CONSTANT)
        op_ret = pad_op(self.image_nds, top_pads, bottom_pads, left_pads, right_pads)
        self._helper(op_ret)

        pad_script_op = matx.script(
            vision.PadWithBorderOp)(
            device=self.device,
            pad_values=(114, 114, 114),
            pad_type=vision.BORDER_CONSTANT)

        script_op_ret = pad_script_op(self.image_nds, top_pads, bottom_pads, left_pads, right_pads)
        self._helper(script_op_ret)

    def test_thread_pool_is_nullptr(self):
        os.environ["MATX_NUM_GTHREADS"] = "0"
        pad_op = vision.PadOp(device=self.device, size=(
            380, 680), pad_values=(114, 114, 114), pad_type=vision.BORDER_CONSTANT)
        op_ret = pad_op(self.image_nds)
        self._helper(op_ret)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), self.origin_res[i])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
