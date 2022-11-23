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
import matx
import numpy as np
from matx.vision.tv_transforms import Compose, Decode


script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestDecode(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(script_path, '..', '..', 'data', 'origin_image.jpeg')
        f = open(image_file, "rb")
        self.img_binary = f.read()
        f.close()
        self.img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        self.device_id = 0
        return super().setUp()

    def test_convert_dtype(self):
        bytedvision_op = Compose(0, [Decode(to_rgb=True)])
        bytedvision_res = bytedvision_op([self.img_binary])[0].asnumpy()
        assert bytedvision_res.shape == self.img.shape

    def test_scripted_convert_dtype(self):
        op = matx.script(Decode)(to_rgb=True)
        composed_op = matx.script(Compose)(0, [op])
        bytedvision_res = composed_op([self.img_binary])[0].asnumpy()
        assert bytedvision_res.shape == self.img.shape


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
