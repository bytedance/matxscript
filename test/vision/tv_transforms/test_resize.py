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
from matx.vision import SYNC
from torchvision import transforms
from matx.vision.tv_transforms import Compose, Resize, RandomResizedCrop


script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestResize(unittest.TestCase):
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
    """
    def test_resize(self):
        size = [200]
        bytedvision_op = Compose(0, [Resize(size, device_id=self.device_id, sync=SYNC)])
        bytedvision_res = bytedvision_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.Resize(size)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        assert bytedvision_res.shape == torchvision_res.shape
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res)

    def test_scripted_resize(self):
        size = [200]
        op = matx.script(Resize)(size, device_id=self.device_id, sync=SYNC)
        composed_op = matx.script(Compose)(0, [op])
        bytedvision_res = composed_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.Resize(size)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        assert bytedvision_res.shape == torchvision_res.shape
        np.testing.assert_almost_equal(bytedvision_res, torchvision_res)
    """


class TestRandomResizedCrop(unittest.TestCase):
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

    def test_random_resize_crop(self):
        size = [400]
        bytedvision_op = Compose(0, [RandomResizedCrop(size, device_id=self.device_id, sync=SYNC)])
        bytedvision_res = bytedvision_op([self.img_nd, self.img_nd])[0].asnumpy()
        torchvision_op = transforms.RandomResizedCrop(size)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        assert bytedvision_res.shape == torchvision_res.shape

    def test_scripted_random_resize_crop(self):
        size = [400]
        op = matx.script(RandomResizedCrop)(size, device_id=self.device_id, sync=SYNC)
        composed_op = matx.script(Compose)(0, [op])
        bytedvision_res = composed_op([self.img_nd])[0].asnumpy()
        torchvision_op = transforms.RandomResizedCrop(size)
        torchvision_res = np.array(torchvision_op(self.img_tensor))
        assert bytedvision_res.shape == torchvision_res.shape


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
