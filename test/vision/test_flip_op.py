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
from matx.vision import HORIZONTAL_FLIP, FlipOp

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestFlipOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.flip_code = HORIZONTAL_FLIP
        self.origin_res = cv2.flip(image, self.flip_code)
        self.image_nd = matx.array.from_numpy(image)
        self.device = matx.Device("cpu")

        return super().setUp()

    def test_flip_op(self):
        flip_op = FlipOp(self.device, self.flip_code)
        op_ret = flip_op([self.image_nd])
        self._helper(op_ret[0])

        op_ret = flip_op([self.image_nd], [self.flip_code])
        self._helper(op_ret[0])

    def test_scripted_flip_op(self):
        script_flip_op = matx.script(
            FlipOp)(
            self.device,
            self.flip_code)
        script_ret = script_flip_op([self.image_nd])
        self._helper(script_ret[0])

        script_ret = script_flip_op([self.image_nd], [self.flip_code])
        self._helper(script_ret[0])

    def _helper(self, ret):
        np.testing.assert_almost_equal(ret.asnumpy(), self.origin_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
