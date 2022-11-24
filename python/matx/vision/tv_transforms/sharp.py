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

from typing import Any, List
import sys
matx = sys.modules['matx']
from .. import ASYNC, BORDER_REPLICATE, Conv2dOp

from ._base import BaseInterfaceClass, BatchRandomBaseClass
from ._common import _assert


class RandomAdjustSharpness(BaseInterfaceClass):
    def __init__(self,
                 sharpness_factor: float,
                 p: float = 0.5,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self.p: float = p
        self.sharpness_factor: float = sharpness_factor

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomAdjustSharpnessImpl(device, device_str, self.sharpness_factor, self.p, sync)


class RandomAdjustSharpnessImpl(BatchRandomBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 sharpness_factor: float,
                 p: float = 0.5,
                 sync: int = ASYNC) -> None:
        self.device_str: str = device_str
        self.p: float = p
        self.sync: int = sync
        super().__init__(prob=self.p)
        _assert(
            sharpness_factor >= 0,
            "sharpness_factor ({}) is not non-negative.".format(sharpness_factor))
        self.sharpness_factor: float = sharpness_factor

        edge_value: float = (1 - sharpness_factor) / 13.0
        center_value: float = (1 - sharpness_factor) * 5.0 / 13.0 + sharpness_factor
        self.kernel: List[List[float]] = [[edge_value] * 3,
                                          [edge_value, center_value, edge_value], [edge_value] * 3]
        self.op: Conv2dOp = Conv2dOp(device, BORDER_REPLICATE)
        self.sync: int = sync
        self.name: str = "RandomAdjustSharpness"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        batch_size: int = len(imgs)
        kernels: List[List[List[float]]] = [self.kernel for _ in range(batch_size)]
        return self.op(imgs, kernels, sync=self.sync)

    def __repr__(self) -> str:
        return self.name + '(sharpness_factor={}, p={}, device={}, sync={})'.format(
            self.sharpness_factor, self.p, self.device_str, self.sync)
