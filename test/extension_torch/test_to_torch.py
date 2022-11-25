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
import numpy as np
import torch
import matx


class TestToTorch(unittest.TestCase):
    def test_cpu_copy(self):
        nd = matx.NDArray([1, 2, 3, 4], [2, 2], "int32")
        t = nd.torch()
        self.assertEqual(t.dtype, torch.int32)
        self.assertEqual(t.device, torch.device("cpu"))
        self.assertEqual(np.sum(t.numpy() - nd.numpy()), 0)
        t[0][0] = 2
        self.assertEqual(nd[0][0], 1)

    # def test_cuda_copy(self):
    #    nd = matx.NDArray([1, 2, 3, 4], [2, 2], "int32", "cuda:0")
    #    t = nd.torch()
    #    self.assertEqual(t.dtype, torch.int32)
    #    self.assertEqual(t.device, torch.device("cuda:0"))
    #    self.assertEqual(np.sum(t.cpu().numpy() - nd.numpy()), 0)
    #    t[0][0] = 2
    #    self.assertEqual(nd.numpy()[0][0], 1)

    def test_cpu(self):
        nd = matx.NDArray([1, 2, 3, 4], [2, 2], "int32")
        t = nd.torch(copy=False)
        self.assertEqual(t.dtype, torch.int32)
        self.assertEqual(t.device, torch.device("cpu"))
        self.assertEqual(np.sum(t.numpy() - nd.numpy()), 0)
        t[0][0] = 2
        self.assertEqual(nd[0][0], 2)

    # def test_cuda(self):
    #    nd = matx.NDArray([1, 2, 3, 4], [2, 2], "int32", "cuda:0")
    #    t = nd.torch(copy=False)
    #    self.assertEqual(t.dtype, torch.int32)
    #    self.assertEqual(t.device, torch.device("cuda:0"))
    #    self.assertEqual(np.sum(t.cpu().numpy() - nd.numpy()), 0)
    #    t[0][0] = 2
    #    self.assertEqual(nd.numpy()[0][0], 2)

    def test_cpu_copy_non_continuous(self):
        arr = matx.NDArray(1, [8, 2, 8], "int32")
        arr = arr[2:8:2]
        print(arr.is_contiguous())
        t = arr.torch(copy=True)
        self.assertEqual(t.dtype, torch.int32)
        self.assertEqual(t.device, torch.device("cpu"))
        self.assertTrue(np.alltrue(t.numpy() == arr.numpy()))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
