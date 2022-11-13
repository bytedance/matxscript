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
import shutil
import numpy as np
import torch
import matx

from matx.extension.pytorch.pytorch_module import PytorchModule


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


class TestPytorchModule(unittest.TestCase):
    def test_torch_infer(self):
        my_cell = MyCell()
        script_model = torch.jit.trace(my_cell, (torch.rand(3, 4), torch.rand(3, 4)))
        save_path = "test_torch_infer.jit"
        script_model.save(save_path)

        torch_model = PytorchModule(location=save_path, device=-1)
        torch_op = torch_model.make_pipeline_op()
        x = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))
        h = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))

        def process(x, h):
            h1, h2 = torch_op(x, h)
            return h1, h2

        data = process(x, h)
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))
        print(data[0])

        module_jit = matx.pipeline.Trace(process, x, h)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))
        print(data[0])

        save_path = "test_location_saved_model"
        module_jit.save(save_path)
        module_jit = matx.pipeline.Load(save_path, device=-1)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))
        print(data[0])

    def test_torch_model(self):
        my_cell = MyCell()
        torch_model = PytorchModule(model=my_cell, device=-1)
        torch_op = torch_model.make_pipeline_op()
        x = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))
        h = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))

        def process(x, h):
            h1, h2 = torch_op(x, h)
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

        save_path = "test_model_saved_model"
        module_jit.save(save_path)
        module_jit = matx.pipeline.Load(save_path, device=-1)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, tuple))
        self.assertTrue(isinstance(data[0], matx.array.NDArray))
        self.assertSequenceEqual(data[0].shape(), (3, 4))
        print(data[0])

    def test_func_trace(self):
        my_cell = MyCell()
        my_cell.eval()
        for param in my_cell.parameters():
            param.requires_grad = False

        def get_trace_func(model):
            def fn(*args):
                a, b = model(*args)
                return a

            return fn

        trace_func = get_trace_func(my_cell)
        torch_model = PytorchModule(model=trace_func, device=-1)
        torch_op = torch_model.make_pipeline_op()

        def process(x, h):
            return torch_op(x, h)

        x = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))
        h = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))

        data = process(x, h)
        self.assertTrue(isinstance(data, matx.array.NDArray))
        self.assertSequenceEqual(data.shape(), (3, 4))

        module_jit = matx.pipeline.Trace(process, x, h)
        data = module_jit.run({'x': x, 'h': h})
        self.assertTrue(isinstance(data, matx.array.NDArray))
        self.assertSequenceEqual(data.shape(), (3, 4))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
