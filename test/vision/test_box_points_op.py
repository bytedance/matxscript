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


class TestBoxPointsOp(unittest.TestCase):
    def setUp(self) -> None:
        self.device = matx.Device("cpu")

        image_file = os.path.join(script_path, '..', 'data', 'origin_image.jpeg')
        image = cv2.imread(image_file)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.rects = [cv2.minAreaRect(c) for c in contours]

    def test_simple_box_points_rect(self):
        op = vision.BoxPointsOp(self.device)
        for rect in self.rects:
            cv2_box = cv2.boxPoints(rect)
            matx_box = op(rect)
            np.testing.assert_almost_equal(cv2_box, matx_box)

    def test_simple_scripted_box_points_rect(self):
        op = matx.script(vision.BoxPointsOp)(self.device)
        for rect in self.rects:
            cv2_box = cv2.boxPoints(rect)
            matx_box = op(rect)
            np.testing.assert_almost_equal(cv2_box, matx_box)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
