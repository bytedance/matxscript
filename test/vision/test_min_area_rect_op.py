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

from difflib import diff_bytes
import os
import unittest
import cv2
import numpy as np
import matx
from matx import vision

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestMinAreaRectOp(unittest.TestCase):
    def setUp(self) -> None:
        self.device = matx.Device("cpu")

        image_file = os.path.join(script_path, '..', 'data', 'origin_image.jpeg')
        image = cv2.imread(image_file)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)

        self.contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def compare_rect(self, rect1, rect2):
        def float_eq(f1, f2, threshold=0.9):
            return abs(f1 - f2) < threshold

        if rect1 == rect2:
            return
        center1 = rect1[0]
        center2 = rect2[0]
        size1 = rect1[1]
        size2 = rect2[1]
        angle1 = rect1[2]
        angle2 = rect2[2]
        if not (float_eq(center1[0], center2[0]) and float_eq(center1[1], center2[1])):
            self.assertTrue(False, f"center not equal {rect1} vs {rect2}")
        if float_eq(
                size1[0],
                size1[1],
                0.1) and float_eq(
                size1[0],
                size2[0],
                0.1) and float_eq(
                size2[0],
                size2[1],
                0.1):
            if not (float_eq(abs(angle1 - angle2), 180, 5) or (float_eq(abs(angle1 - angle2),
                    90, 5) or float_eq(abs(angle1 - angle2), 270), 5)):
                self.assertTrue(
                    False, f"angle not equal (even after rotating 90 180 and 270 degree) {rect1} vs {rect2}")
        elif float_eq(size1[0], size2[0], 0.1) and float_eq(size1[1], size2[1], 0.1):
            if not float_eq(abs(angle1 - angle2), 180, 5):
                self.assertTrue(
                    False, f"angle not equal (even after rotating 180 degree) {rect1} vs {rect2}")
        elif float_eq(size1[0], size2[1], 0.1) and float_eq(size1[1], size2[0], 0.1):
            if not (float_eq(abs(angle1 - angle2), 90, 5)
                    or float_eq(abs(angle1 - angle2), 270), 5):
                self.assertTrue(
                    False, f"angle not equal (even after rotating 90 and 270 degree) {rect1} vs {rect2}")
        else:
            self.assertTrue(False, f"size not equal {rect1} vs {rect2}")

    def compare_rect_not_assert(self, rect1, rect2):
        def float_eq(f1, f2, threshold=0.9):
            return abs(f1 - f2) < threshold

        if rect1 == rect2:
            return
        center1 = rect1[0]
        center2 = rect2[0]
        size1 = rect1[1]
        size2 = rect2[1]
        angle1 = rect1[2]
        angle2 = rect2[2]
        if not (float_eq(center1[0], center2[0]) and float_eq(center1[1], center2[1])):
            return False
        if float_eq(
                size1[0],
                size1[1],
                0.1) and float_eq(
                size1[0],
                size2[0],
                0.1) and float_eq(
                size2[0],
                size2[1],
                0.1):
            if not (float_eq(abs(angle1 - angle2), 180, 5) or (float_eq(abs(angle1 - angle2),
                    90, 5) or float_eq(abs(angle1 - angle2), 270), 5)):
                return False
        elif float_eq(size1[0], size2[0], 0.1) and float_eq(size1[1], size2[1], 0.1):
            if not float_eq(abs(angle1 - angle2), 180, 5):
                return False
        elif float_eq(size1[0], size2[1], 0.1) and float_eq(size1[1], size2[0], 0.1):
            if not (float_eq(abs(angle1 - angle2), 90, 5)
                    or float_eq(abs(angle1 - angle2), 270), 5):
                return False
        else:
            return False
        return True

    def test_simple_min_area_rect(self):
        op = vision.MinAreaRectOp(self.device)
        print(len(self.contours))
        n_diff = 0
        for contour in self.contours:
            cv2_rect = cv2.minAreaRect(contour)  # tuple(tuple('center'), tuple('size'), 'angle')
            matx_rect = op(matx.array.from_numpy(contour))
            if not (self.compare_rect_not_assert(cv2_rect, matx_rect)
                    ):  # the version of matx opencv may be diff from python opencv's
                n_diff += 1
        self.assertLessEqual(n_diff, 150)

    def test_scripted_simple_min_area_rect(self):
        op = matx.script(vision.MinAreaRectOp)(self.device)
        print(len(self.contours))
        n_diff = 0
        for contour in self.contours:
            cv2_rect = cv2.minAreaRect(contour)  # tuple(tuple('center'), tuple('size'), 'angle')
            matx_rect = op(matx.array.from_numpy(contour))
            if not (self.compare_rect_not_assert(cv2_rect, matx_rect)
                    ):  # the version of matx opencv may be diff from python opencv's
                n_diff += 1
        self.assertLessEqual(n_diff, 150)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
