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
import numpy as np
import torch
import matx
import random

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


class TestPyTorchJitFile(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_path = SCRIPT_PATH + "/../tempdir/"
        self.data_path = SCRIPT_PATH + "/../data/"
        self.work_path = self.tmp_path + "TestPyTorchJitFile/"
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)

    def test_script_jit(self):
        tmp_dir = self.work_path + os.sep + str(random.randint(1, 100000000))
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        jit_loc = tmp_dir + os.sep + './my_cell.jit'
        with torch.no_grad():
            my_cell = torch.jit.script(MyCell())
            my_cell.save(jit_loc)
        torch_model = matx.script(jit_loc, backend="PyTorch", device=-1)
        x = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))
        h = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))

        def process(x, h):
            h1, h2 = torch_model(x, h)
            return h1, h2

        data = process(x, h)
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))

        module_jit = matx.pipeline.Trace(process, x, h)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))

        save_path = tmp_dir + os.sep + "matx_torch_model"
        module_jit.save(save_path)
        module_jit = matx.pipeline.Load(save_path, device=-1)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))
        print(data[0])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
