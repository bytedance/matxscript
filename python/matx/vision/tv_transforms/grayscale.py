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
from .. import ASYNC, COLOR_RGB2GRAY, CvtColorOp, ChannelReorderOp

from ._base import BatchBaseClass, BatchRandomBaseClass, BaseInterfaceClass
from ._common import get_image_num_channels


class Grayscale(BaseInterfaceClass):
    def __init__(self,
                 num_output_channels: int = 1,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        assert num_output_channels in [1, 3], "Number of output channels should be 1 or 3."
        self._num_output_channels: int = num_output_channels

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return GrayscaleImpl(device, device_str, self._num_output_channels, sync)


class GrayscaleImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 num_output_channels: int,
                 sync: int) -> None:
        super().__init__()
        self.device_str: str = device_str
        self.num_output_channels: int = num_output_channels
        self.gray_op: CvtColorOp = CvtColorOp(device, COLOR_RGB2GRAY)
        self.channel_reorder_op: ChannelReorderOp = ChannelReorderOp(device)
        self.sync: int = sync
        self.name: str = "GrayscaleImpl"

    def _process_to_one_channel(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.gray_op(imgs, sync=self.sync)

    def _process_to_three_channel(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        orders = [[0] * 3 for _ in range(len(imgs))]
        imgs = self.gray_op(imgs)
        return self.channel_reorder_op(imgs, orders, self.sync)

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        if self.num_output_channels == 1:
            return self._process_to_one_channel(imgs)
        return self._process_to_three_channel(imgs)

    def __repr__(self) -> str:
        return self.name + '(num_output_channels={}, device={}, sync={})'.format(
            self.num_output_channels, self.device_str, self.sync)


class RandomGrayscale(BaseInterfaceClass):
    def __init__(self,
                 p: float = 0.1,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        self._p: float = p
        super().__init__(device_id=device_id, sync=sync)
        assert 0 <= self._p <= 1, "Probablity should be between 0 and 1."

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomGrayscaleImpl(device, device_str, self._p, sync)


class RandomGrayscaleImpl(BatchRandomBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 p: float,
                 sync: int = ASYNC) -> None:
        self.device_str: str = device_str
        self.p: float = p
        self.sync: int = sync
        super().__init__(prob=self.p)
        self.gray_op: CvtColorOp = CvtColorOp(device, COLOR_RGB2GRAY)
        self.channel_reorder_op: ChannelReorderOp = ChannelReorderOp(device)
        self.name: str = "RandomGrayscaleImpl"

    def _process_to_one_channel(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.gray_op(imgs, sync=self.sync)

    def _process_to_three_channel(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        orders = [[0] * 3 for _ in range(len(imgs))]
        imgs = self.gray_op(imgs)
        return self.channel_reorder_op(imgs, orders, self.sync)

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        num_output_channels = get_image_num_channels(imgs[0])
        if num_output_channels == 1:
            return self._process_to_one_channel(imgs)
        return self._process_to_three_channel(imgs)

    def __repr__(self) -> str:
        return self.name + '(p={}, device={}, sync={})'.format(
            self.p, self.device_str, self.sync)
