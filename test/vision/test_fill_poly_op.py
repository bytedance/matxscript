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


class TestFillPolyOp(unittest.TestCase):
    def setUp(self) -> None:
        self.device = matx.Device("cpu")

        image_file = os.path.join(script_path, '..', 'data', 'origin_image.jpeg')
        self.image = cv2.imread(image_file)
        self.matx_image = matx.array.from_numpy(self.image)

    def test_fill_poly_1(self):
        op = vision.FillPolyOp(self.device)
        color = (255, 255, 255)
        points = np.array([[16, 13], [35, 13], [25, 30]], dtype="int32")
        cv2_result = cv2.fillPoly(self.image, [points], color)
        matx_result = op(self.matx_image, [matx.array.from_numpy(points)], color)
        np.testing.assert_almost_equal(matx_result.asnumpy(), cv2_result)

    def test_script_fill_poly_1(self):
        op = matx.script(vision.FillPolyOp)(self.device)
        color = (255, 255, 255)
        points = np.array([[16, 13], [35, 13], [25, 30]], dtype="int32")
        cv2_result = cv2.fillPoly(self.image, [points], color)
        matx_result = op(self.matx_image, [matx.array.from_numpy(points)], color)
        np.testing.assert_almost_equal(matx_result.asnumpy(), cv2_result)

    def test_fill_poly_2(self):
        op = vision.FillPolyOp(self.device)
        color = (255, 255, 10)
        line_type = 8
        shift = 0
        offset = (0, 0)
        points = np.array([[16, 13], [35, 13], [25, 30]], dtype="int32")
        cv2_result = cv2.fillPoly(self.image, [points], color, line_type, shift, offset)
        matx_result = op(
            self.matx_image, [
                matx.array.from_numpy(points)], color, line_type, shift, offset)
        np.testing.assert_almost_equal(matx_result.asnumpy(), cv2_result)

    def test_fill_poly_3(self):
        op = vision.FillPolyOp(self.device)
        color = (10,)
        line_type = cv2.LINE_4
        shift = 0
        offset = (0, 0)
        points = np.array([[16, 13], [35, 13], [25, 30]], dtype="int32")
        cv2_result = cv2.fillPoly(self.image, [points], color, line_type, shift, offset)
        matx_result = op(
            self.matx_image, [
                matx.array.from_numpy(points)], color, line_type, shift, offset)
        np.testing.assert_almost_equal(matx_result.asnumpy(), cv2_result)

    def test_fill_poly_4(self):
        op = vision.FillPolyOp(self.device)
        color = (10,)
        line_type = cv2.LINE_4
        shift = 5
        offset = (0, 0)
        points = np.array([[16, 13], [35, 13], [25, 30]], dtype="int32")
        cv2_result = cv2.fillPoly(self.image, [points], color, line_type, shift, offset)
        matx_result = op(
            self.matx_image, [
                matx.array.from_numpy(points)], color, line_type, shift, offset)
        np.testing.assert_almost_equal(matx_result.asnumpy(), cv2_result)

    def test_fill_poly_5(self):
        op = vision.FillPolyOp(self.device)
        color = (10,)
        line_type = cv2.LINE_8
        shift = 5
        offset = (-1, 10)
        points = np.array([[16, 13], [35, 13], [25, 30]], dtype="int32")
        cv2_result = cv2.fillPoly(self.image, [points], color, line_type, shift, offset)
        matx_result = op(
            self.matx_image, [
                matx.array.from_numpy(points)], color, line_type, shift, offset)
        np.testing.assert_almost_equal(matx_result.asnumpy(), cv2_result)

    def test_script_fill_poly_5(self):
        op = matx.script(vision.FillPolyOp)(self.device)
        color = (10,)
        line_type = cv2.LINE_8
        shift = 5
        offset = (-1, 10)
        points = np.array([[16, 13], [35, 13], [25, 30]], dtype="int32")
        cv2_result = cv2.fillPoly(self.image, [points], color, line_type, shift, offset)
        matx_result = op(
            self.matx_image, [
                matx.array.from_numpy(points)], color, line_type, shift, offset)
        np.testing.assert_almost_equal(matx_result.asnumpy(), cv2_result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
