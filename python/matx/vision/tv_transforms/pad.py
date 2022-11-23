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
from .. import ASYNC, PadWithBorderOp
from ._base import BaseInterfaceClass, BatchBaseClass
from ._common import _torch_padding_mode, _assert


class Pad(BaseInterfaceClass):
    def __init__(self,
                 padding: List[int],
                 fill: List[int] = [0],
                 padding_mode: str = "constant",
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self._padding: List[int] = padding
        self._fill: List[int] = fill
        self._padding_mode: str = padding_mode

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return PadImpl(device, device_str, self._padding, self._fill, self._padding_mode, sync)


class PadImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 padding: List[int],
                 fill: List[int] = [0],
                 padding_mode: str = "constant",
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.device_str: str = device_str

        if len(padding) == 1:
            self.padding: List[int] = [padding[0]] * 4
        elif len(padding) == 2:
            self.padding: List[int] = [padding[0], padding[1], padding[0], padding[1]]
        elif len(padding) == 4:  # left, top, right, bottom
            self.padding: List[int] = [padding[0], padding[1], padding[2], padding[3]]
        else:
            _assert(
                False,
                "Padding must be an int or a 1, 2, or 4 element tuple, not a {} element tuple".format(
                    len(padding)))

        if len(fill) == 1:
            self.fill: Tuple[int, int, int] = (fill[0], fill[0], fill[0])
        else:
            self.fill: Tuple[int, int, int] = (fill[0], fill[1], fill[2])

        #_assert(padding_mode in _torch_padding_mode_list, "padding_mode not found")

        self.padding_mode: str = _torch_padding_mode(padding_mode)
        self.sync: int = sync
        self.op: PadWithBorderOp = PadWithBorderOp(device, self.fill, self.padding_mode)
        self.name: str = "Pad"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(
            imgs, [
                self.padding[1]], [
                self.padding[3]], [
                self.padding[0]], [
                    self.padding[2]], sync=self.sync)

    def __repr__(self) -> str:
        return self.name + '(padding={0}, fill={1}, padding_mode={2}, device={3}, sync={4})'.\
            format(self.padding, self.fill, self.padding_mode, self.device_str, self.sync)
