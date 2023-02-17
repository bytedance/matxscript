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
from matx import vision as byted_vision
import matx
import numpy as np
os.environ["MATX_NUM_GTHREADS"] = "1"
script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestImencodeOp(unittest.TestCase):

    def _helper(self, bv_out):
        for i in range(3):
            np.testing.assert_almost_equal(
                bv_out[i], self.images[i])

    def setUp(self) -> None:
        image_file1 = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')
        image_file2 = os.path.join(
            script_path, '..', 'data', 'example.jpeg')
        image_file3 = os.path.join(
            script_path, '..', 'data', 'exif_orientation5.jpg')
        
        image1 = cv2.imread(image_file1)
        image2 = cv2.imread(image_file2)
        image3 = cv2.imread(image_file3)
        self.images = [image1, image2, image3]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in self.images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

        
    def test_BGR(self):
        op = byted_vision.ImencodeOp(self.device, "BGR", 100, False)
        #r = op(self.image_nd)
        print("hi")
        #out1 = cv2.imdecode(r[0].asnumpy(), cv2.IMREAD_COLOR)
        #out2 = cv2.imdecode(r[1].asnumpy(), cv2.IMREAD_COLOR)
        #out3 = cv2.imdecode(r[2].asnumpy(), cv2.IMREAD_COLOR)
        #self._helper([out1, out2, out3])
        
"""
    def test_scripted_BGR(self):
        op = matx.script(byted_vision.ImencodeOp)(self.device, "BGR", 100, False)
        r = op(self.image_nd)
        out1 = cv2.imdecode(r[0].asnumpy(), cv2.IMREAD_COLOR)
        out2 = cv2.imdecode(r[1].asnumpy(), cv2.IMREAD_COLOR)
        out3 = cv2.imdecode(r[2].asnumpy(), cv2.IMREAD_COLOR)
        self._helper([out1, out2, out3])

    def test_RGB(self):
        op = byted_vision.ImencodeOp(self.device, "RGB", 100, False)
        r = op(self.image_nd)

        out1 = cv2.cvtColor(cv2.imdecode(r[0].asnumpy(), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
        out2 = cv2.cvtColor(cv2.imdecode(r[1].asnumpy(), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
        out3 = cv2.cvtColor(cv2.imdecode(r[2].asnumpy(), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
        self._helper([out1, out2, out3])
"""



if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
