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
import random

script_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


class TestCudaConv2dOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file).astype("float32")
        images = [image, image, image]
        kernel_sizes = [[3, 3], [1, 3], [5, 3]]
        self.batch_size = len(images)
        self.kernels = [
            self.create_random_kernel(
                kernel_sizes[i]) for i in range(
                self.batch_size)]

        self.origin_res = [
            cv2.filter2D(images[i], -1, np.array(self.kernels[i])) for i in range(self.batch_size)]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def create_random_kernel(self, ksize):
        kernel = []
        for row in range(ksize[1]):
            tmp_kernel = []
            for col in range(ksize[0]):
                tmp_kernel.append(random.random())
            kernel.append(tmp_kernel)
        return kernel

    def test_cuda_conv2d_op(self):
        conv2d_op = byted_vision.Conv2dOp(self.device)
        op_ret = conv2d_op(self.image_nd, self.kernels)
        self._helper(op_ret)

    def test_scripted_cuda_conv2d_op(self):
        script_conv2d_op = matx.script(byted_vision.Conv2dOp)(self.device)
        script_ret = script_conv2d_op(self.image_nd, self.kernels)
        self._helper(script_ret)

    def _cuda_conv2d_cpu_input_sync(self, op):
        op_ret = op(self.image_nd_cpu, self.kernels, [], byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret)

    def test_cuda_conv2d_cpu_input_sync(self):
        conv2d_op = byted_vision.Conv2dOp(self.device)
        self._cuda_conv2d_cpu_input_sync(conv2d_op)

    def test_cuda_conv2d_cpu_input_sync_scripted(self):
        conv2d_op = matx.script(byted_vision.Conv2dOp)(self.device)
        self._cuda_conv2d_cpu_input_sync(conv2d_op)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), self.origin_res[i])


class TestCudaSharpenOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file).astype("float32")
        images = [image, image, image]
        self.batch_size = len(images)
        self.alpha = [0.1, 0.5, 0.7]
        self.lightness = [0.8, 1.0, 1.2]
        kernels = [
            self.create_sharpen_kernel(
                self.alpha[i],
                self.lightness[i]) for i in range(
                self.batch_size)]

        self.origin_res = [
            cv2.filter2D(images[i], -1, kernels[i]) for i in range(self.batch_size)]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def create_sharpen_kernel(self, alpha, lightness):
        kernel = []
        for _ in range(4):
            kernel.append(-alpha)
        kernel.append(1 + 7 * alpha + lightness * alpha)
        for _ in range(4):
            kernel.append(-alpha)
        return np.array(kernel).reshape(3, 3)

    def test_cuda_sharpen_op(self):
        op = byted_vision.SharpenOp(self.device)
        op_ret = op(self.image_nd, self.alpha, self.lightness)
        self._helper(op_ret)

    def test_scripted_cuda_sharpen_op(self):
        script_op = matx.script(byted_vision.SharpenOp)(self.device)
        script_ret = script_op(self.image_nd, self.alpha, self.lightness)
        self._helper(script_ret)

    def _cuda_sharpen_cpu_input_sync(self, op):
        op_ret = op(self.image_nd_cpu, self.alpha, self.lightness, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret)

    def test_cuda_sharpen_cpu_input_sync(self):
        op = byted_vision.SharpenOp(self.device)
        self._cuda_sharpen_cpu_input_sync(op)

    def test_cuda_sharpen_cpu_input_sync_scripted(self):
        op = matx.script(byted_vision.SharpenOp)(self.device)
        self._cuda_sharpen_cpu_input_sync(op)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), self.origin_res[i])


class TestCudaEmbossOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file).astype("float32")
        images = [image, image, image]
        self.batch_size = len(images)
        self.alpha = [0.1, 0.5, 0.7]
        self.strength = [0.2, 0.5, 1.0]
        kernels = [
            self.create_emboss_kernel(
                self.alpha[i],
                self.strength[i]) for i in range(
                self.batch_size)]

        self.origin_res = [
            cv2.filter2D(images[i], -1, kernels[i]) for i in range(self.batch_size)]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def create_emboss_kernel(self, alpha, strength):
        tmp = alpha * strength
        kernel = [-alpha - tmp, -tmp, 0, -tmp, 1, tmp, 0, tmp, alpha + tmp]
        return np.array(kernel).reshape(3, 3)

    def test_cuda_emboss_op(self):
        op = byted_vision.EmbossOp(self.device)
        op_ret = op(self.image_nd, self.alpha, self.strength)
        self._helper(op_ret)

    def test_scripted_cuda_emboss_op(self):
        script_op = matx.script(byted_vision.EmbossOp)(self.device)
        script_ret = script_op(self.image_nd, self.alpha, self.strength)
        self._helper(script_ret)

    def _cuda_emboss_cpu_input_sync(self, op):
        op_ret = op(self.image_nd_cpu, self.alpha, self.strength, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret)

    def test_cuda_emboss_cpu_input_sync(self):
        op = byted_vision.EmbossOp(self.device)
        self._cuda_emboss_cpu_input_sync(op)

    def test_cuda_emboss_cpu_input_sync_scripted(self):
        op = matx.script(byted_vision.EmbossOp)(self.device)
        self._cuda_emboss_cpu_input_sync(op)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), self.origin_res[i])


class TestCudaEdgeDetectOp(unittest.TestCase):
    def setUp(self) -> None:
        image_file = os.path.join(
            script_path, '..', 'data', 'origin_image.jpeg')

        image = cv2.imread(image_file).astype("float32")
        images = [image, image, image]
        self.batch_size = len(images)
        self.alpha = [0.1, 0.5, 0.7]
        kernels = [
            self.create_edge_detect_kernel(
                self.alpha[i]) for i in range(
                self.batch_size)]

        self.origin_res = [
            cv2.filter2D(images[i], -1, kernels[i]) for i in range(self.batch_size)]

        self.image_nd = [matx.array.from_numpy(i, "gpu:0") for i in images]
        self.image_nd_cpu = [matx.array.from_numpy(i) for i in images]
        self.device = matx.Device("gpu:0")

        return super().setUp()

    def create_edge_detect_kernel(self, alpha):
        kernel = [0, alpha, 0, alpha, 1 - alpha * 5, alpha, 0, alpha, 0]
        return np.array(kernel).reshape(3, 3)

    def test_cuda_edge_detect_op(self):
        op = byted_vision.EdgeDetectOp(self.device)
        op_ret = op(self.image_nd, self.alpha)
        self._helper(op_ret)

    def test_scripted_cuda_edge_detect_op(self):
        script_op = matx.script(byted_vision.EdgeDetectOp)(self.device)
        script_ret = script_op(self.image_nd, self.alpha)
        self._helper(script_ret)

    def _cuda_edge_detect_cpu_input_sync(self, op):
        op_ret = op(self.image_nd_cpu, self.alpha, byted_vision.SYNC_CPU)
        for nd in op_ret:
            self.assertEqual(nd.device(), "cpu")
        self._helper(op_ret)

    def test_cuda_edge_detect_cpu_input_sync(self):
        op = byted_vision.EdgeDetectOp(self.device)
        self._cuda_edge_detect_cpu_input_sync(op)

    def test_cuda_edge_detect_cpu_input_sync_scripted(self):
        op = matx.script(byted_vision.EdgeDetectOp)(self.device)
        self._cuda_edge_detect_cpu_input_sync(op)

    def _helper(self, ret):
        for i in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret[i].asnumpy(), self.origin_res[i])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
