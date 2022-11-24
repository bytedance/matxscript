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
from .. import ASYNC, ImdecodeOp

from ._base import BaseInterfaceClass, BatchBaseClass


class Decode(BaseInterfaceClass):
    def __init__(self,
                 to_rgb: bool = False,
                 pool_size: int = 8,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self.to_rgb: bool = to_rgb
        self.pool_size: int = pool_size

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return DecodeImpl(device, device_str, self.to_rgb, self.pool_size, sync)


class DecodeImpl:
    def __init__(self,
                 device: Any,
                 device_str: str,
                 to_rgb: bool = False,
                 pool_size: int = 8,
                 sync: int = ASYNC):
        self.device: Any = device
        self.device_str: str = device_str
        self.color_format: str = "RGB" if to_rgb else "BGR"
        self.pool_size: int = pool_size
        self.op: ImdecodeOp = ImdecodeOp(self.device, self.color_format, self.pool_size)
        self.sync: int = sync
        self.name: str = "Decode"

    def _process(self, imgs: List[bytes]) -> List[matx.NDArray]:
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return '{}(format={}, pool_size={}, device={}, sync={})'.format(
            self.name, self.color_format, self.pool_size, self.device_str, self.sync)

    def __call__(
            self,
            images: List[bytes],
            apply_index: List[int] = []) -> List[matx.NDArray]:
        assert len(apply_index) == 0, "apply_index is not support by jpeg decode."
        new_images: List[matx.NDArray] = self._process(images)
        return new_images
