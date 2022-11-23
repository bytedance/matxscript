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

import numbers
from typing import Tuple, Sequence, Any, Union, Dict, List
import sys
matx = sys.modules['matx']
from .. import BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101, BORDER_REFLECT
from .. import INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
import random
import torch

cpu_device: Any = matx.Device("cpu")
gpu_device_0: Any = matx.Device("gpu:0")
gpu_device_1: Any = matx.Device("gpu:1")
gpu_device_2: Any = matx.Device("gpu:2")
gpu_device_3: Any = matx.Device("gpu:3")
gpu_device_4: Any = matx.Device("gpu:4")
gpu_device_5: Any = matx.Device("gpu:5")
gpu_device_6: Any = matx.Device("gpu:6")
gpu_device_7: Any = matx.Device("gpu:7")


class DeviceManagerOp:
    def __init__(self) -> None:
        self.devices: Dict[int, Any] = {}
        self.devices[0] = gpu_device_0
        self.devices[1] = gpu_device_1
        self.devices[2] = gpu_device_2
        self.devices[3] = gpu_device_3
        self.devices[4] = gpu_device_4
        self.devices[5] = gpu_device_5
        self.devices[6] = gpu_device_6
        self.devices[7] = gpu_device_7
        self.devices[-1] = cpu_device

    def __call__(self, device_id: int) -> Any:
        return self.devices[device_id]


DeviceManager = matx.script(DeviceManagerOp)()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(message)
        assert False, ""


def _setup_size(size: List[int]) -> Tuple[int, int]:
    assert len(size) == 1 or len(size) == 2, "Kernel size should be a tuple/list of two integers."
    if len(size) == 1:
        return (size[0], size[0])
    return (size[0], size[1])


def _check_sequence_input(x: List[Any], name: str, req_sizes: List[int]) -> None:
    msg: str = str(req_sizes[0]) if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x: List[float], name: str, req_sizes: List[int] = [2]) -> List[float]:
    if len(x) == 1:
        if x[0] < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x[0], x[0]]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


def get_image_num_channels(img: matx.NDArray) -> int:
    shape = img.shape()
    if len(shape) < 2:
        assert False, "Input image should have shape (h, w) or (h, w, c), but get only 1 dim"
    if len(shape) == 2:
        return 1
    return shape[2]


def create_device(device_id):
    if device_id == -2:  # -1 means use cpu, so here use -2 to indicate not set
        # get device from torch
        device_id = torch.cuda.current_device()
    device_str = "gpu:{}".format(device_id) if device_id >= 0 else "cpu"
    device = matx.Device(device_str)
    return device_str, device


def _uniform_random(rmin: float, rmax: float, size: int) -> List[float]:
    rrange: float = rmax - rmin
    rvalues = []
    for i in range(size):
        tmp_value: float = random.random() * rrange + rmin
        rvalues.append(tmp_value)
    return rvalues


def _randint(rmin: int, rmax: int, size: int) -> List[int]:
    rrange: int = rmax - rmin
    rvalues = []
    for i in range(size):
        tmp_value: int = int(random.random() * rrange + rmin)
        if tmp_value >= rmax:
            tmp_value = rmax
        rvalues.append(tmp_value)
    return rvalues


def _torch_padding_mode(p: str) -> str:
    if p == "constant":
        return BORDER_CONSTANT
    elif p == "edge":
        return BORDER_REPLICATE
    elif p == "reflect":
        return BORDER_REFLECT_101
    elif p == "symmetric":
        return BORDER_REFLECT
    else:
        _assert(False, "padding_mode not found")
        return ""


_torch_interp_mode = {
    "nearest": INTER_NEAREST,
    "bilinear": INTER_LINEAR,
    "bicubic": INTER_CUBIC,
    "lanczos": INTER_LANCZOS4
}
