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


class TestWarpPerspectiveOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        images = [image, image, image]
        height, width = image.shape[:2]
        self.batch_size = len(images)
        self.interp = byted_vision.INTER_LINEAR
        self.border_type = byted_vision.BORDER_CONSTANT
        self.border_value = (255, 255, 255)
        src1_ptrs = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        dst1_ptrs = [(0, height * 0.13), (width * 0.9, 0),
                     (width * 0.2, height * 0.7), (width * 0.8, height - 1)]

        src2_ptrs = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        dst2_ptrs = [(0, height * 0.03), (width * 0.93, 0),
                     (width * 0.23, height * 0.73), (width * 0.83, height - 1)]

        src3_ptrs = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        dst3_ptrs = [(0, height * 0.33), (width * 0.73, 0),
                     (width * 0.23, height * 0.83), (width * 0.63, height - 1)]
        self.pts = [[src1_ptrs, dst1_ptrs], [src2_ptrs, dst2_ptrs], [src3_ptrs, dst3_ptrs]]

        self.origin_res = []
        for i in range(self.batch_size):
            cur_src_pts = np.float32([[i[0], i[1]] for i in self.pts[i][0]])
            cur_dst_pts = np.float32([[i[0], i[1]] for i in self.pts[i][1]])
            matrix = cv2.getPerspectiveTransform(cur_src_pts, cur_dst_pts)
            tmp_res = cv2.warpPerspective(
                image, matrix, (width, height), borderValue=self.border_value)
            self.origin_res.append(tmp_res)

        self.image_nd = [matx.array.from_numpy(i) for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("cpu")

        return super().setUp()

    def test_warp_perspective_op(self):
        op = byted_vision.WarpPerspectiveOp(self.device, pad_values=self.border_value)
        op_ret = op(self.image_nd, self.pts)
        self._helper(op_ret)

    def test_scripted_warp_perspective_op(self):
        script_op = matx.script(
            byted_vision.WarpPerspectiveOp)(
            self.device,
            pad_values=self.border_value)
        script_ret = script_op(self.image_nd, self.pts)
        self._helper(script_ret)

    def _helper(self, ret):
        for i in range(self.batch_size):
            res = ret[i].asnumpy()
            diff = np.abs(res - self.origin_res[i])
            diff_ratio = np.sum(diff > 0) / diff.size
            if diff_ratio > 0.05:
                if cv2.__version__.startswith("3.4"):
                    np.testing.assert_almost_equal(res, self.origin_res[i])
                else:
                    print(
                        "[warning] the diff ratio in cpu warp perspective op is greater than 0.05. "
                        f"This is not a failure because your opencv version is {cv2.__version__}, while matx uses 3.4.8.")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
