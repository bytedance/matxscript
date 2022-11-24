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


class ConvertImageDtype(BaseInterfaceClass):
    def __init__(self,
                 dtype: str,
                 global_scale: float = 1.0,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self.dtype: str = dtype
        self.global_scale: float = global_scale

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return ConvertImageDtypeImpl(device, device_str, self.dtype, self.global_scale, sync)


class ConvertImageDtypeImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 dtype: str,
                 global_scale: float = 1.0,
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.device: Any = device
        self.device_str: str = device_str
        self.dtype: str = dtype
        self.global_scale: float = global_scale
        self.sync: int = sync
        self.op: NormalizeOp = NormalizeOp(self.device,
                                           mean=[0.0, 0.0, 0.0],
                                           std=[1.0, 1.0, 1.0],
                                           global_scale=self.global_scale, dtype=dtype)
        self.name: str = "ConvertImageDtype"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return '{}(dtype={}, scale={}, device={}, sync={})'.format(
            self.name, self.dtype, self.global_scale, self.device_str, self.sync)
