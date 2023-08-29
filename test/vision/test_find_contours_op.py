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


class TestFindContoursOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)
        self.thresh = thresh
        self.thresh_nd = matx.array.from_numpy(thresh)
        self.device = matx.Device("cpu")
        return super().setUp()

    def contours_eq(self, matx_contours, opencv_contours):
        self.assertEqual(len(matx_contours), len(opencv_contours))
        for matx_ctr, opencv_ctr in zip(matx_contours, opencv_contours):
            np.testing.assert_equal(matx_ctr.asnumpy(), opencv_ctr)

    def hierarchy_eq(self, matx_hierarchy, opencv_hierarchy):
        np.testing.assert_equal(matx_hierarchy.asnumpy(), opencv_hierarchy)

    def test_find_contours0(self):
        cv2_contours, cv2_hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        find_contours = vision.FindContoursOp(self.device)
        matx_contours, matx_hierarchy = find_contours(
            self.thresh_nd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_eq(matx_contours, cv2_contours)
        self.hierarchy_eq(matx_hierarchy, cv2_hierarchy)

    def test_find_contours1(self):
        cv2_contours, cv2_hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        find_contours = vision.FindContoursOp(self.device)
        matx_contours, matx_hierarchy = find_contours(
            self.thresh_nd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_eq(matx_contours, cv2_contours)
        self.hierarchy_eq(matx_hierarchy, cv2_hierarchy)

    def test_find_contours2(self):
        cv2_contours, cv2_hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        find_contours = vision.FindContoursOp(self.device)
        matx_contours, matx_hierarchy = find_contours(
            self.thresh_nd, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_eq(matx_contours, cv2_contours)
        self.hierarchy_eq(matx_hierarchy, cv2_hierarchy)

    def test_find_contours3(self):
        cv2_contours, cv2_hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        find_contours = vision.FindContoursOp(self.device)
        matx_contours, matx_hierarchy = find_contours(
            self.thresh_nd, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_eq(matx_contours, cv2_contours)
        self.hierarchy_eq(matx_hierarchy, cv2_hierarchy)

    def test_find_contours4(self):
        cv2_contours, cv2_hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        find_contours = vision.FindContoursOp(self.device)
        matx_contours, matx_hierarchy = find_contours(
            self.thresh_nd, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        self.contours_eq(matx_contours, cv2_contours)
        self.hierarchy_eq(matx_hierarchy, cv2_hierarchy)

    def test_scripted_find_contours(self):
        cv2_contours, cv2_hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        find_contours = matx.script(vision.FindContoursOp)(self.device)
        matx_contours, matx_hierarchy = find_contours(
            self.thresh_nd, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.contours_eq(matx_contours, cv2_contours)
        self.hierarchy_eq(matx_hierarchy, cv2_hierarchy)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
