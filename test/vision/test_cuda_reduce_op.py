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
import unittest
import cv2
import numpy as np
import matx
from matx import vision as byted_vision

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestSumOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.batch_size = 2
        self.image_nd = [
            matx.array.from_numpy(
                image, "gpu:0"), matx.array.from_numpy(
                image, "gpu:0")]
        self.device = matx.Device("gpu:0")
        self.per_channel_origin_res = np.sum(image, axis=(0, 1))
        self.origin_res = np.sum(self.per_channel_origin_res)

        return super().setUp()

    def test_sum_op(self):
        sum_op = byted_vision.SumOp(self.device)
        op_ret = sum_op(self.image_nd)
        per_channel_sum_op = byted_vision.SumOp(self.device, True)
        per_channel_op_ret = per_channel_sum_op(self.image_nd)
        self._helper(op_ret, per_channel_op_ret)

    def test_scripted_sum_op(self):
        script_sum_op = matx.script(byted_vision.SumOp)(self.device)
        script_ret = script_sum_op(self.image_nd)
        per_channel_sum_op = matx.script(byted_vision.SumOp)(self.device, True)
        per_channel_script_ret = per_channel_sum_op(self.image_nd)
        self._helper(script_ret, per_channel_script_ret)

    def _cuda_sum_sync_cpu(self, op, per_channel_op):
        op_ret = op(self.image_nd, byted_vision.SYNC_CPU)
        per_channel_op_ret = per_channel_op(self.image_nd, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret, per_channel_op_ret)

    def test_sum_sync_cpu(self):
        op = byted_vision.SumOp(self.device)
        per_channel_op = byted_vision.SumOp(self.device, True)
        self._cuda_sum_sync_cpu(op, per_channel_op)

    def test_sum_sync_cpu_scripted(self):
        op = matx.script(byted_vision.SumOp)(self.device)
        per_channel_op = matx.script(byted_vision.SumOp)(self.device, True)
        self._cuda_sum_sync_cpu(op, per_channel_op)

    def _helper(self, ret, per_channel_ret):
        res = ret.asnumpy()
        per_channel_res = per_channel_ret.asnumpy()
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(res[i][0], self.origin_res)
            np.testing.assert_almost_equal(per_channel_res[i], self.per_channel_origin_res)


class TestMeanOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file)
        self.batch_size = 2
        self.image_nd = [
            matx.array.from_numpy(
                image, "gpu:0"), matx.array.from_numpy(
                image, "gpu:0")]
        self.device = matx.Device("gpu:0")
        self.per_channel_origin_res = np.mean(image, axis=(0, 1))
        self.origin_res = np.mean(self.per_channel_origin_res)

        return super().setUp()

    def test_mean_op(self):
        mean_op = byted_vision.MeanOp(self.device)
        op_ret = mean_op(self.image_nd)
        per_channel_op = byted_vision.MeanOp(self.device, True)
        per_channel_op_ret = per_channel_op(self.image_nd)
        self._helper(op_ret, per_channel_op_ret)

    def test_scripted_mean_op(self):
        script_mean_op = matx.script(byted_vision.MeanOp)(self.device)
        script_ret = script_mean_op(self.image_nd)
        per_channel_op = matx.script(byted_vision.MeanOp)(self.device, True)
        per_channel_script_ret = per_channel_op(self.image_nd)
        self._helper(script_ret, per_channel_script_ret)

    def _cuda_mean_sync_cpu(self, op, per_channel_op):
        op_ret = op(self.image_nd, byted_vision.SYNC_CPU)
        per_channel_op_ret = per_channel_op(self.image_nd, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret, per_channel_op_ret)

    def test_mean_sync_cpu(self):
        op = byted_vision.MeanOp(self.device)
        per_channel_op = byted_vision.MeanOp(self.device, True)
        self._cuda_mean_sync_cpu(op, per_channel_op)

    def test_mean_sync_cpu_scripted(self):
        op = matx.script(byted_vision.MeanOp)(self.device)
        per_channel_op = matx.script(byted_vision.MeanOp)(self.device, True)
        self._cuda_mean_sync_cpu(op, per_channel_op)

    def _helper(self, ret, per_channel_ret):
        res = ret.asnumpy()
        per_channel_res = per_channel_ret.asnumpy()
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(res[i][0], self.origin_res)
            np.testing.assert_almost_equal(per_channel_res[i], self.per_channel_origin_res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
