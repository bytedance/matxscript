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
from re import S
import unittest
import cv2
import numpy as np
import matx
from matx import vision

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestEstimateAffinePartial2DOp(unittest.TestCase):
    def setUp(self) -> None:
        self.device = matx.Device("cpu")
        return super().setUp()

    def test_eap2d_1(self):
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        op = vision.EstimateAffinePartial2DOp(self.device)
        cv2_m, cv2_inliers = cv2.estimateAffinePartial2D(pts1, pts2)
        matx_m, matx_inliers = op(matx.array.from_numpy(pts1), matx.array.from_numpy(pts2))
        np.testing.assert_almost_equal(cv2_m, matx_m.asnumpy())
        np.testing.assert_array_equal(cv2_inliers, matx_inliers.asnumpy())

    def test_scripted_eap2d_1(self):
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        op = matx.script(vision.EstimateAffinePartial2DOp)(self.device)
        cv2_m, cv2_inliers = cv2.estimateAffinePartial2D(pts1, pts2)
        matx_m, matx_inliers = op(matx.array.from_numpy(pts1), matx.array.from_numpy(pts2))
        np.testing.assert_almost_equal(cv2_m, matx_m.asnumpy())
        np.testing.assert_array_equal(cv2_inliers, matx_inliers.asnumpy())

    def test_eap2d_LMEDS(self):
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        op = vision.EstimateAffinePartial2DOp(self.device)
        cv2_m, cv2_inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.LMEDS)
        matx_m, matx_inliers = op(matx.array.from_numpy(
            pts1), matx.array.from_numpy(pts2), method=cv2.LMEDS)
        np.testing.assert_almost_equal(cv2_m, matx_m.asnumpy())
        np.testing.assert_array_equal(cv2_inliers, matx_inliers.asnumpy())

    def test_eap2d_threshold10(self):
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        op = vision.EstimateAffinePartial2DOp(self.device)
        cv2_m, cv2_inliers = cv2.estimateAffinePartial2D(pts1, pts2, ransacReprojThreshold=10)
        matx_m, matx_inliers = op(matx.array.from_numpy(
            pts1), matx.array.from_numpy(pts2), ransacReprojThreshold=10)
        np.testing.assert_almost_equal(cv2_m, matx_m.asnumpy())
        np.testing.assert_array_equal(cv2_inliers, matx_inliers.asnumpy())

    def test_eap2d_10_iter1000_confidence0d5_refineliter2(self):
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        op = vision.EstimateAffinePartial2DOp(self.device)
        cv2_m, cv2_inliers = cv2.estimateAffinePartial2D(
            pts1, pts2, maxIters=1000, confidence=0.5, refineIters=2)
        matx_m, matx_inliers = op(matx.array.from_numpy(pts1), matx.array.from_numpy(
            pts2), maxIters=1000, confidence=0.5, refineIters=2)
        np.testing.assert_almost_equal(cv2_m, matx_m.asnumpy())
        np.testing.assert_array_equal(cv2_inliers, matx_inliers.asnumpy())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
