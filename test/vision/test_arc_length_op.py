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


class TestArcLengthOp(unittest.TestCase):
    def setUp(self) -> None:
        self.device = matx.Device("cpu")
        image_file = os.path.join(script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, self.thresh = cv2.threshold(imgray, 127, 255, 0)
        self.thresh_nd = matx.array.from_numpy(self.thresh)
        self.contours, self.hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        find_contours = matx.script(vision.FindContoursOp)(self.device)
        self.matx_contours, self.matx_hierarchy = find_contours(
            self.thresh_nd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return super().setUp()

    def test_arc_length_use_cv_contours(self):
        arc_length_op = vision.ArcLengthOp(self.device)
        for i in range(len(self.contours)):
            opencv_perimeter = cv2.arcLength(self.contours[i], True)
            matx_perimeter = arc_length_op(matx.array.from_numpy(self.contours[i]), True)
            self.assertEqual(opencv_perimeter, matx_perimeter)

    def test_scripted_arc_length_use_cv_contours(self):
        arc_length_op = matx.script(vision.ArcLengthOp)(self.device)
        for i in range(len(self.contours)):
            opencv_perimeter = cv2.arcLength(self.contours[i], True)
            matx_perimeter = arc_length_op(matx.array.from_numpy(self.contours[i]), True)
            self.assertEqual(opencv_perimeter, matx_perimeter)

    def test_arc_length_use_matx_contours(self):
        arc_length_op = vision.ArcLengthOp(self.device)
        for i in range(len(self.contours)):
            opencv_perimeter = cv2.arcLength(self.matx_contours[i].asnumpy(), False)
            matx_perimeter = arc_length_op(self.matx_contours[i], False)
            self.assertEqual(opencv_perimeter, matx_perimeter)

    def test_scripted_arc_length_use_matx_contours(self):
        arc_length_op = matx.script(vision.ArcLengthOp)(self.device)
        for i in range(len(self.contours)):
            opencv_perimeter = cv2.arcLength(self.matx_contours[i].asnumpy(), False)
            matx_perimeter = arc_length_op(self.matx_contours[i], False)
            self.assertEqual(opencv_perimeter, matx_perimeter)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
