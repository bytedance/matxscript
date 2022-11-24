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
from matx.vision import ImdecodeOp, SYNC_CPU
import numpy as np
os.environ["MATX_NUM_GTHREADS"] = "1"
script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestImdecodeOp(unittest.TestCase):
    def setUp(self) -> None:
        self.image_file1 = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')
        self.image_file2 = os.path.join(
            script_path, '..', 'data', 'example.jpeg')
        with open(self.image_file1, "rb") as fd:
            self.image_content1 = fd.read()
        with open(self.image_file2, "rb") as fd:
            self.image_content2 = fd.read()
        self.device = matx.Device("cpu")

    def test_BGR(self):
        op = ImdecodeOp(self.device, "BGR")
        r = op([self.image_content1, self.image_content2, self.image_content1])
        cv_out1 = cv2.imread(self.image_file1)
        cv_out2 = cv2.imread(self.image_file2)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)

    def test_scripted_BGR(self):
        op = matx.script(ImdecodeOp)(self.device, "BGR")
        r = op([self.image_content1, self.image_content2, self.image_content1])
        cv_out1 = cv2.imread(self.image_file1)
        cv_out2 = cv2.imread(self.image_file2)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)

    def test_RGB(self):
        op = ImdecodeOp(self.device, "RGB")
        r = op([self.image_content1, self.image_content2, self.image_content1])
        cv_out1 = cv2.cvtColor(cv2.imread(self.image_file1), cv2.COLOR_BGR2RGB)
        cv_out2 = cv2.cvtColor(cv2.imread(self.image_file2), cv2.COLOR_BGR2RGB)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)

    def test_scripted_RGB(self):
        op = matx.script(ImdecodeOp)(self.device, "RGB")
        r = op([self.image_content1, self.image_content2, self.image_content1])
        cv_out1 = cv2.cvtColor(cv2.imread(self.image_file1), cv2.COLOR_BGR2RGB)
        cv_out2 = cv2.cvtColor(cv2.imread(self.image_file2), cv2.COLOR_BGR2RGB)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)

    def test_to_cpu(self):
        op = ImdecodeOp(self.device, "BGR")
        r = op([self.image_content1, self.image_content2,
               self.image_content1], SYNC_CPU)
        self.assertEqual(r[0].device(), "cpu")
        self.assertEqual(r[1].device(), "cpu")
        cv_out1 = cv2.imread(self.image_file1)
        cv_out2 = cv2.imread(self.image_file2)
        self._helper(r[0], cv_out1)
        self._helper(r[1], cv_out2)
        self._helper(r[2], cv_out1)

    def _helper(self, nd_out, cv_out):
        self.assertEqual(np.sum(nd_out.asnumpy() - cv_out), 0)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
