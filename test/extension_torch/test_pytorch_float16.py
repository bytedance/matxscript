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
        xf = x.to(dtype=torch.float32)
        hf = h.to(dtype=torch.float32)
        new_h = self.linear(xf) + hf
        return new_h.to(dtype=torch.float16), new_h.to(dtype=torch.float16)


class TestPytorchFloat16(unittest.TestCase):

    def test_float16(self):
        my_cell = torch.jit.script(MyCell())
        torch_model = matx.script(my_cell)
        x = matx.array.from_numpy(np.random.rand(3, 4).astype("float16"))
        h = matx.array.from_numpy(np.random.rand(3, 4).astype("float16"))

        def process(x, h):
            h1, h2 = torch_model(x, h)
            return h1, h2

        data = process(x, h)
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))

        module_jit = matx.trace(process, x, h)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))

        save_path = "test_model_saved_model"
        module_jit.save(save_path)
        module_jit = matx.load(save_path, device=-1)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))
        print(data[0])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
