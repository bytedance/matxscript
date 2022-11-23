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

from typing import Any, Tuple, List
import sys
matx = sys.modules['matx']
from .. import ASYNC, NormalizeOp

from ._base import BaseInterfaceClass, BatchBaseClass


class Normalize(BaseInterfaceClass):
    def __init__(self,
                 mean: List[float],
                 std: List[float],
                 global_scale: float = 1.0,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self._mean: List[float] = mean
        self._std: List[float] = std
        self._global_scale: float = global_scale

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return NormalizeImpl(device, device_str, self._mean, self._std, self._global_scale, sync)


class NormalizeImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 mean: List[float],
                 std: List[float],
                 global_scale: float = 1.0,
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.global_scale: float = global_scale
        self.mean: List[float] = [i / global_scale for i in mean]
        self.std: List[float] = [i / global_scale for i in std]
        self.device_str: str = device_str
        self.op: NormalizeOp = NormalizeOp(device, self.mean, self.std)
        self.sync: int = sync
        self.name: str = "Normalize"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return self.name + '(mean={0}, std={1}, device={2}, sync={3})'.format(
            self.mean, self.std, self.device_str, self.sync)
