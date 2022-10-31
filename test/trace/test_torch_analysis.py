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
import torch
import matx
import numpy as np


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


x = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))
h = matx.array.from_numpy(np.random.rand(3, 4).astype("float32"))


def mock_an_model(save_path):
    my_cell = MyCell()
    torch_model = matx.script(my_cell)

    def process(x, h):
        h1, h2 = torch_model(x, h)
        return h1, h2

    module_jit = matx.trace(process, x, h)
    module_jit.save(save_path)


def get_torch_info(save_path):
    module_jit = matx.load(save_path, device=-1)
    # the input info can be found in warmup data
    step_data = module_jit.gen_step_meta({'x': x, 'h': h})

    torch_info = []
    for op in step_data["ops"]:
        if op["op_cls"] == "PyTorchInferOp":
            op_inputs = op["inputs"]
            op_output = op["output"]
            op_impl = op["attributes"]["impl"]
            assert op_impl["op_cls"] == "TorchInferOp"
            op_device = op_impl["attributes"]["device"]
            op_output_to_cpu = op_impl["attributes"]["output_to_cpu"]
            op_model = op_impl["attributes"]["model"]
            model_info = module_jit.get_nested_op_attributes("TorchModel", op_model)
            jit_location = model_info["location"]
            torch_info.append(
                {
                    "inputs": op_inputs,
                    "output": op_output,
                    "device": op_device,
                    "output_to_cpu": op_output_to_cpu,
                    "location": jit_location,
                }
            )
    return torch_info


if __name__ == "__main__":
    save_path = "./test_model_saved_model"
    mock_an_model(save_path)
    torch_infos = get_torch_info(save_path)
    for ti in torch_infos:
        print(ti)
