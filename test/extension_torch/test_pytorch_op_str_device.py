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


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h


class TestPyTorchOpStrDevice(unittest.TestCase):
    def test_script_module_with_device(self):
        np_x = np.random.rand(3, 4).astype("float32")
        np_h = np.random.rand(3, 4).astype("float32")
        th_x = torch.from_numpy(np_x)
        th_h = torch.from_numpy(np_h)
        tx_x = matx.array.from_numpy(np_x)
        tx_h = matx.array.from_numpy(np_h)

        my_cell = MyCell()
        th_res = my_cell(th_x, th_h)
        th_res = th_res.detach().numpy()

        torch_model = matx.script(my_cell, device="cpu:0")

        def process(x, h):
            new_h = torch_model(x, h)
            return new_h

        data = process(tx_x, tx_h)
        data = data.numpy()
        self.assertTrue(np.alltrue(np.isclose(data, th_res)))

        module_jit = matx.trace(process, tx_x, tx_h)
        data = module_jit(x=tx_x, h=tx_h)
        data = data.numpy()
        self.assertTrue(np.alltrue(np.isclose(data, th_res)))

        save_path = "TestPyTorchOpStrDevice_test_script_module_with_device"
        module_jit.save(save_path)
        module_jit = matx.load(save_path, device="cpu:0")
        data = module_jit(x=tx_x, h=tx_h)
        data = data.numpy()
        self.assertTrue(np.alltrue(np.isclose(data, th_res)))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
