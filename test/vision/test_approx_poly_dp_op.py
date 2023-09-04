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


class TestApproxPolyDPOp(unittest.TestCase):
    def setUp(self) -> None:
        self.device = matx.Device("cpu")

        image_file = os.path.join(script_path, '..', 'data', 'origin_image.jpeg')
        image = cv2.imread(image_file)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)

        self.contours, self.hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.perimeters = [cv2.arcLength(cts, True) for cts in self.contours]

        return super().setUp()

    def test_approx_poly_dp_contours(self):
        approx_poly_dp = vision.ApproxPolyDPOp(self.device)
        for i in range(len(self.contours)):
            count = self.contours[i]
            epsilon = 0.01 * self.perimeters[i]
            cv2_approximations = cv2.approxPolyDP(count, epsilon, True)
            matx_approximations = approx_poly_dp(matx.array.from_numpy(count), epsilon, True)
            np.testing.assert_equal(matx_approximations.asnumpy(), cv2_approximations)

    def test_approx_poly_dp_contours_not_closed(self):
        approx_poly_dp = vision.ApproxPolyDPOp(self.device)
        for i in range(len(self.contours)):
            count = self.contours[i]
            epsilon = 0.01 * self.perimeters[i]
            cv2_approximations = cv2.approxPolyDP(count, epsilon, False)
            matx_approximations = approx_poly_dp(matx.array.from_numpy(count), epsilon, False)
            np.testing.assert_equal(matx_approximations.asnumpy(), cv2_approximations)

    def test_approx_poly_dp_contours(self):
        approx_poly_dp = matx.script(vision.ApproxPolyDPOp)(self.device)
        for i in range(len(self.contours)):
            count = self.contours[i]
            epsilon = 0.01 * self.perimeters[i]
            cv2_approximations = cv2.approxPolyDP(count, epsilon, True)
            matx_approximations = approx_poly_dp(matx.array.from_numpy(count), epsilon, True)
            np.testing.assert_equal(matx_approximations.asnumpy(), cv2_approximations)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
