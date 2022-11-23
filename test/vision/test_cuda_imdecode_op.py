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


class TestImdecodeOp(unittest.TestCase):
    def setUp(self) -> None:
        self.image_file1 = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')
        self.image_file2 = os.path.join(
            script_path, '..', 'data', 'example.jpeg')
        self.image_file3 = os.path.join(
            script_path, '..', 'data', 'exif_orientation5.jpg')
        with open(self.image_file1, "rb") as fd:
            self.image_content1 = fd.read()
        with open(self.image_file2, "rb") as fd:
            self.image_content2 = fd.read()
        with open(self.image_file3, "rb") as fd:
            self.image_content3 = fd.read()
        self.device = matx.Device("cuda:0")

        self.no_jpeg_file = os.path.join(
            script_path, "..", "data", "origin_image.sm.webp")
        with open(self.no_jpeg_file, "rb") as fd:
            self.no_jpeg_content = fd.read()

    def test_BGR(self):
        op = byted_vision.ImdecodeOp(self.device, "BGR")
        r = op([self.image_content1, self.image_content2,
               self.image_content1, self.no_jpeg_content,
               self.image_content3])
        cv_out1 = cv2.imread(self.image_file1)
        cv_out2 = cv2.imread(self.image_file2)
        cv_out3 = cv2.imread(self.image_file3)
        cv_nojpeg = cv2.imread(self.no_jpeg_file)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)
        # self._helper(r[4], cv_out3) # do not check exif for now

        # test fall back to cpu
        self.assertEqual(np.sum(r[3].asnumpy() - cv_nojpeg), 0)

    def test_scripted_BGR(self):
        op = matx.script(byted_vision.ImdecodeOp)(self.device, "BGR")
        r = op([self.image_content1, self.image_content2,
               self.image_content1, self.no_jpeg_content,
               self.image_content3])
        cv_out1 = cv2.imread(self.image_file1)
        cv_out2 = cv2.imread(self.image_file2)
        cv_out3 = cv2.imread(self.image_file3)
        cv_nojpeg = cv2.imread(self.no_jpeg_file)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)
        # self._helper(r[4], cv_out3) # do not check exif for now

        # test fall back to cpu
        self.assertEqual(np.sum(r[3].asnumpy() - cv_nojpeg), 0)

    def test_RGB(self):
        op = byted_vision.ImdecodeOp(self.device, "RGB")
        r = op([self.image_content1, self.image_content2,
               self.image_content1, self.no_jpeg_content,
               self.image_content3])
        cv_out1 = cv2.cvtColor(cv2.imread(self.image_file1), cv2.COLOR_BGR2RGB)
        cv_out2 = cv2.cvtColor(cv2.imread(self.image_file2), cv2.COLOR_BGR2RGB)
        cv_out3 = cv2.cvtColor(cv2.imread(self.image_file3), cv2.COLOR_BGR2RGB)
        cv_nojpeg = cv2.cvtColor(cv2.imread(
            self.no_jpeg_file), cv2.COLOR_BGR2RGB)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)
        # self._helper(r[4], cv_out3) # do not check exif for now

        # test fall back to cpu
        self.assertEqual(np.sum(r[3].asnumpy() - cv_nojpeg), 0)

    def test_scripted_RGB(self):
        op = matx.script(byted_vision.ImdecodeOp)(self.device, "RGB")
        r = op([self.image_content1, self.image_content2,
               self.image_content1, self.no_jpeg_content,
               self.image_content3])
        cv_out1 = cv2.cvtColor(cv2.imread(self.image_file1), cv2.COLOR_BGR2RGB)
        cv_out2 = cv2.cvtColor(cv2.imread(self.image_file2), cv2.COLOR_BGR2RGB)
        cv_out3 = cv2.cvtColor(cv2.imread(self.image_file3), cv2.COLOR_BGR2RGB)
        cv_nojpeg = cv2.cvtColor(cv2.imread(
            self.no_jpeg_file), cv2.COLOR_BGR2RGB)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)
        # self._helper(r[4], cv_out3) # do not check exif for now
        # test fall back to cpu
        self.assertEqual(np.sum(r[3].asnumpy() - cv_nojpeg), 0)

    def test_exception(self):
        op = byted_vision.ImdecodeOp(self.device, "BGR")
        content = b'\xe4\xb8\xaa\xe4\xbd\x93\xe5\xaf\xbf\xe5\x91\xbd\xe7\x9a\x84\xe5\xbb\xb6\xe9\x95\xbf\xe6\x98\xaf\xe6\x96\x87\xe6\x98\x8e\xe6\xad\xa5\xe5\x85\xa5\xe8\x80\x81\xe5\xb9\xb4\xe7\x9a\x84\xe7\xac\xac\xe4\xb8\x80\xe4\xb8\xaa\xe6\xa0\x87\xe5\xbf\x97'
        self.assertRaises(Exception, op, [content, self.image_content1])

    def test_noexception(self):
        op = byted_vision.ImdecodeNoExceptionOp(self.device, "BGR")
        r, flags = op([self.image_content1, self.image_content2,
                       self.image_content1, self.no_jpeg_content,
                       self.image_content3])
        cv_out1 = cv2.imread(self.image_file1)
        cv_out2 = cv2.imread(self.image_file2)
        cv_out3 = cv2.imread(self.image_file3)
        cv_nojpeg = cv2.imread(self.no_jpeg_file)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)
        # self._helper(r[4], cv_out3) # do not check exif for now
        # test fall back to cpu
        self.assertEqual(np.sum(r[3].asnumpy() - cv_nojpeg), 0)
        self.assertSequenceEqual([1] * 5, flags)

    def test_noexception2(self):
        op = byted_vision.ImdecodeNoExceptionOp(self.device, "BGR")
        content = b'\xe4\xb8\xaa\xe4\xbd\x93\xe5\xaf\xbf\xe5\x91\xbd\xe7\x9a\x84\xe5\xbb\xb6\xe9\x95\xbf\xe6\x98\xaf\xe6\x96\x87\xe6\x98\x8e\xe6\xad\xa5\xe5\x85\xa5\xe8\x80\x81\xe5\xb9\xb4\xe7\x9a\x84\xe7\xac\xac\xe4\xb8\x80\xe4\xb8\xaa\xe6\xa0\x87\xe5\xbf\x97'
        images, flags = op([self.image_content1, content])
        cv_out1 = cv2.imread(self.image_file1)
        self._helper(images[0], cv_out1)
        self.assertSequenceEqual([1, 0], flags)

    def _helper(self, nd_out, cv_out):
        self.assertEqual(nd_out.asnumpy().shape, cv_out.shape)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
