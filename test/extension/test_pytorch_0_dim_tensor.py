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


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


class TestPytorch_0_Dim_Tensor(unittest.TestCase):

    def test_0_dim_tensor(self):
        th_model = MyNet()
        tx_model = matx.script(th_model, device=-1)
        my_input = matx.NDArray([2.0], [], "float32")

        def my_process(q):
            r = tx_model(q)
            return r

        print(my_process(my_input))

        jit_mod = matx.trace(my_process, my_input)
        print(jit_mod.run({"q": my_input}))
        my_input = matx.NDArray([2.0, 3.0], [], "float32")
        print(jit_mod.run({"q": my_input}))
        my_input = matx.NDArray([[2.0, 3.0]], [], "float32")
        print(jit_mod.run({"q": my_input}))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
