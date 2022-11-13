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
import numpy as np
import torch
import matx
from typing import Any, List


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


# make torch op
my_cell = MyCell()
global_torch_op = matx.script(my_cell)


class MyGenerator:

    def __init__(self, torch_op: Any, h: matx.NDArray):
        self.torch_op: Any = torch_op
        self.h: matx.NDArray = h

    def __call__(self, x: matx.NDArray) -> List[matx.NDArray]:
        results = [global_torch_op(x, self.h)]
        for i in range(10):
            results.append(self.torch_op(x, self.h))
        return results


def test_control_flow():
    # make examples
    x = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))
    h = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))

    # make general python op
    my_gen_op = matx.script(MyGenerator)(torch_op=global_torch_op, h=h)

    def workflow(x):
        return my_gen_op(x)

    # just run
    data = workflow(x)
    print(data)

    # trace
    jit_mod = matx.trace(workflow, x)
    data = jit_mod.run({'x': x})
    print(data)

    # save
    save_path = './test_torch_control_flow'
    jit_mod.save(save_path)
    jit_mod = matx.load(save_path, device='cpu')
    data = jit_mod.run({'x': x})
    print(data)


if __name__ == "__main__":
    test_control_flow()
