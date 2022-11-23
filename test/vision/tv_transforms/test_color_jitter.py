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
import io
from PIL import Image
import cv2
import numpy as np
import matx
from torchvision import transforms
from matx.vision import SYNC
from matx.vision.tv_transforms import Compose, ColorJitter


script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestColorJitter(unittest.TestCase):

    def setUp(self) -> None:
        image_file = os.path.join(script_path, '..', '..', 'data', 'origin_image.jpeg')
        img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        f = open(image_file, "rb")
        img_binary = f.read()
        f.close()
        self.device_id = 0
        self.img_nd = matx.array.from_numpy(img, "gpu:{}".format(self.device_id))
        self.img_tensor = Image.open(io.BytesIO(img_binary))
        return super().setUp()

    def test_brightness(self):
        brightness = [1.2, 1.2]
        bytedvision_op = Compose(0, [ColorJitter(brightness=brightness)])
        bytedvision_res = bytedvision_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.ColorJitter(brightness=brightness)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res, decimal=0)

    def test_scripted_brightness(self):
        brightness = [1.2, 1.2]
        op = matx.script(ColorJitter)(brightness=brightness)
        composed_op = matx.script(Compose)(0, [op])
        bytedvision_res = composed_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.ColorJitter(brightness=brightness)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res)

    def test_contrast(self):
        contrast = [0.8, 0.8]
        bytedvision_op = Compose(0, [ColorJitter(
            contrast=contrast,
            device_id=self.device_id,
            sync=SYNC)])
        bytedvision_res = bytedvision_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.ColorJitter(contrast=contrast)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res, decimal=0)

    def test_saturation(self):
        saturation = [1.5, 1.5]
        bytedvision_op = Compose(0, [ColorJitter(
            saturation=saturation,
            device_id=self.device_id,
            sync=SYNC)])
        bytedvision_res = bytedvision_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.ColorJitter(saturation=saturation)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res, decimal=0)

    def test_hue(self):
        hue = [0.05, 0.05]
        bytedvision_op = Compose(0, [ColorJitter(hue=hue, device_id=self.device_id, sync=SYNC)])
        bytedvision_res = bytedvision_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.ColorJitter(hue=hue)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res)

    def test_color_jitter(self):
        brightness = [0.8, 1.2]
        contrast = [0.8, 1.2]
        saturation = [0.8, 1.2]
        hue = [-0.1, 0.1]
        bytedvision_op = Compose(0, [ColorJitter(
            brightness,
            contrast,
            saturation,
            hue,
            device_id=self.device_id,
            sync=SYNC)])
        bytedvision_res = bytedvision_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.ColorJitter(brightness, contrast, saturation, hue)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
