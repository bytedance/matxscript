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


class TestInvertOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.image_nd = matx.array.from_numpy(image, "gpu:0")
        self.image_nd_cpu = matx.array.from_numpy(image)
        self.device = matx.Device("gpu:0")
        self.inverted_image = 255 - image

        return super().setUp()

    def test_invert_op(self):
        invert_op = byted_vision.InvertOp(self.device)
        op_ret = invert_op([self.image_nd])
        self._helper(op_ret[0], self.inverted_image)

    def test_invert_op_per_channel(self):
        invert_op = byted_vision.InvertOp(self.device, per_channel=True)
        op_ret = invert_op([self.image_nd])
        self._helper(op_ret[0], self.inverted_image)

    def test_scripted_invert_op(self):
        script_invert_op = matx.script(byted_vision.InvertOp)(self.device)
        script_ret = script_invert_op([self.image_nd])
        self._helper(script_ret[0], self.inverted_image)

    def test_scripted_invert_op_per_channel(self):
        script_invert_op = matx.script(byted_vision.InvertOp)(
            self.device, per_channel=True)
        script_ret = script_invert_op([self.image_nd])
        self._helper(script_ret[0], self.inverted_image)

    def _invert_op_cpu_input_sync(self, op):
        op_ret = op([self.image_nd_cpu], byted_vision.SYNC_CPU)
        self.assertEqual(op_ret[0].device(), "cpu")
        self._helper(op_ret[0], self.inverted_image)

    def test_invert_op_cpu_input_sync(self):
        invert_op = byted_vision.InvertOp(self.device)
        self._invert_op_cpu_input_sync(invert_op)
        invert_op_per = byted_vision.InvertOp(self.device, per_channel=True)
        self._invert_op_cpu_input_sync(invert_op_per)
        script_invert_op = matx.script(byted_vision.InvertOp)(self.device)
        self._invert_op_cpu_input_sync(script_invert_op)
        script_invert_op_per = matx.script(
            byted_vision.InvertOp)(self.device, per_channel=True)
        self._invert_op_cpu_input_sync(script_invert_op_per)

    def _helper(self, ret, origin_res):
        np.testing.assert_almost_equal(ret.asnumpy(), origin_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
