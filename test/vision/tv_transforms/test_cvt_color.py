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

import unittest
import os
import cv2
import numpy as np
import matx
from torchvision import transforms
from matx.vision import COLOR_BGR2RGB, SYNC
from matx.vision.tv_transforms import Compose, CvtColor


script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestCvtColor(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(script_path, '..', '..', 'data', 'origin_image.jpeg')
        self.device_id = 0
        img = cv2.imread(image_file)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_nd = matx.array.from_numpy(img, "gpu:{}".format(self.device_id))

        return super().setUp()

    def test_cvt_color(self):
        bytedvision_op = Compose(0, [CvtColor(
            COLOR_BGR2RGB,
            device_id=self.device_id,
            sync=SYNC)])
        bytedvision_res = bytedvision_op([self.img_nd])[0].asnumpy()
        np.testing.assert_almost_equal(bytedvision_res, self.img)

    def test_scripted_cvt_color(self):
        bytedvision_op = matx.script(CvtColor)(
            COLOR_BGR2RGB,
            device_id=self.device_id,
            sync=SYNC)
        composed_op = matx.script(Compose)(0, [bytedvision_op])
        bytedvision_res = composed_op([self.img_nd])[0].asnumpy()
        np.testing.assert_almost_equal(bytedvision_res, self.img)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
