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

from typing import Any, Dict, List
import sys
matx = sys.modules['matx']
from .. import ASYNC, HORIZONTAL_FLIP, VERTICAL_FLIP, FlipOp
from ._base import BaseInterfaceClass, BatchRandomBaseClass


class RandomHorizontalFlip(BaseInterfaceClass):
    def __init__(self,
                 p: float = 0.5,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        self._p: float = p
        super().__init__(device_id=device_id, sync=sync)

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomHorizontalFlipImpl(device, device_str, self._p, sync)


class RandomHorizontalFlipImpl(BatchRandomBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 p: float = 0.5,
                 sync: int = ASYNC) -> None:
        self.device_str: str = device_str
        self.p: float = p
        self.sync: int = sync
        super().__init__(prob=self.p)
        flip_code: int = HORIZONTAL_FLIP
        self.op: FlipOp = FlipOp(device, flip_code)
        self.name: str = "RandomHorizontalFlip"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return self.name + \
            '(p={}, device={}, sync={})'.format(self.p, self.device_str, self.sync)


class RandomVerticalFlip(BaseInterfaceClass):
    def __init__(self,
                 p: float = 0.5,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        self._p: float = p
        super().__init__(device_id=device_id, sync=sync)

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomVerticalFlipImpl(device, device_str, self._p, sync)


class RandomVerticalFlipImpl(BatchRandomBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 p: float = 0.5,
                 sync: int = ASYNC) -> None:
        self.device_str: str = device_str
        self.p: float = p
        self.sync: int = sync
        super().__init__(prob=self.p)
        flip_code: int = VERTICAL_FLIP
        self.op: FlipOp = FlipOp(device, flip_code)
        self.name: str = "RandomVerticalFlip"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return self.name + \
            '(p={}, device={}, sync={})'.format(self.p, self.device_str, self.sync)
