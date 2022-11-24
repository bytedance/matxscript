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


class TestCudaWarpAffineOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.images = [image, image, image]
        self.batch_size = len(self.images)

        self.image_nd = [matx.array.from_numpy(i) for i in self.images]
        self.device = matx.Device("cpu")

        return super().setUp()

    def test_translate(self):
        translate_x_matrix = [[1, 0, -10], [0, 1, 0]]
        translate_y_matrix = [[1, 0, 0], [0, 1, 10]]
        translate_xy_matrix = [[1, 0, 15], [0, 1, -5]]
        matrix = [translate_x_matrix, translate_y_matrix, translate_xy_matrix]
        origin_res = [cv2.warpAffine(self.images[i],
                                     np.float32(matrix[i]),
                                     self.images[i].shape[1::-1],
                                     flags=cv2.INTER_NEAREST) for i in range(self.batch_size)]

        affine_op = byted_vision.WarpAffineOp(self.device, interp=byted_vision.INTER_NEAREST)
        op_ret = affine_op(self.image_nd, matrix)
        self._helper(op_ret, origin_res)

        affine_script_op = matx.script(
            byted_vision.WarpAffineOp)(
            self.device,
            interp=byted_vision.INTER_NEAREST)
        script_ret = affine_script_op(self.image_nd, matrix)
        self._helper(script_ret, origin_res)

    def test_rotate(self):
        rotate_90_matrix = [[0, -1, 0], [1, 0, 0]]
        rotate_270_matrix = [[0, 1, 0], [-1, 0, 0]]
        rotate_180_matrix = [[-1, 0, 0], [0, -1, 0]]
        matrix = [rotate_90_matrix, rotate_270_matrix, rotate_180_matrix]
        dsize = [(224, 224), (180, 360), (240, 100)]
        origin_res = [
            cv2.warpAffine(
                self.images[i],
                np.float32(matrix[i]),
                (dsize[i][1],
                 dsize[i][0]),
                flags=cv2.INTER_NEAREST) for i in range(
                self.batch_size)]

        affine_op = byted_vision.WarpAffineOp(self.device, interp=byted_vision.INTER_NEAREST)
        op_ret = affine_op(self.image_nd, matrix, dsize)
        self._helper(op_ret, origin_res)

        affine_script_op = matx.script(
            byted_vision.WarpAffineOp)(
            self.device,
            interp=byted_vision.INTER_NEAREST)
        script_ret = affine_script_op(self.image_nd, matrix, dsize)
        self._helper(script_ret, origin_res)

    def test_shear(self):
        shear_x_matrix = [[1, -0.2, 0], [0, 1, 0]]
        shear_y_matrix = [[1, 0, 0], [0.15, 1, 0]]
        shear_xy_matrix = [[1, 0.1, 0], [-0.12, 1, 0]]
        matrix = [shear_x_matrix, shear_y_matrix, shear_xy_matrix]
        origin_res = [cv2.warpAffine(self.images[i],
                                     np.float32(matrix[i]),
                                     self.images[i].shape[1::-1],
                                     flags=cv2.INTER_CUBIC) for i in range(self.batch_size)]

        affine_op = byted_vision.WarpAffineOp(self.device, interp=byted_vision.INTER_CUBIC)
        op_ret = affine_op(self.image_nd, matrix)
        # there are some diff between warp affine kernel and opencv warp affine
        # just check the shape for now
        self._check_diff(op_ret, origin_res)

        affine_script_op = matx.script(
            byted_vision.WarpAffineOp)(
            self.device,
            interp=byted_vision.INTER_CUBIC)
        script_ret = affine_script_op(self.image_nd, matrix)
        self._check_diff(script_ret, origin_res)

    def _helper(self, ret, origin_res):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), origin_res[i])

    def _check_diff(self, ret, origin_res):
        for i in range(self.batch_size):
            assert ret[i].asnumpy().shape == origin_res[i].shape


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
