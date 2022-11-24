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
from .. import NHWC, NCHW, ASYNC, TransposeOp

from ._base import BaseInterfaceClass


class TransposeCpuOp:
    def __init__(self,
                 device: Any,
                 src_fmt: str = NHWC,
                 dst_fmt: str = NCHW) -> None:
        self.device: Any = device
        self.src_fmt: str = src_fmt
        self.dst_fmt: str = dst_fmt

    def __call__(self, images: matx.NDArray, sync: int = 0) -> matx.NDArray:
        if self.src_fmt == self.dst_fmt:
            return images
        if self.src_fmt == NHWC:
            return images.transpose([0, 3, 1, 2]).contiguous()
        else:
            return images.transpose([0, 2, 3, 1]).contiguous()


class Transpose(BaseInterfaceClass):
    def __init__(self,
                 src_fmt: str = NHWC,
                 dst_fmt: str = NCHW,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self._src_fmt: str = src_fmt
        self._dst_fmt: str = dst_fmt

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return TransposeImpl(device, device_str, self._src_fmt, self._dst_fmt, sync)


class TransposeImpl:
    def __init__(self,
                 device: Any,
                 device_str: str,
                 src_fmt: str = NHWC,
                 dst_fmt: str = NCHW,
                 sync: int = ASYNC) -> None:
        super().__init__()
        if device_str == "cpu":
            self.op: Any = TransposeCpuOp(device, src_fmt, dst_fmt)
        else:
            self.op: Any = TransposeOp(device, src_fmt, dst_fmt)
        self.device_str: str = device_str
        self.src_fmt: str = src_fmt
        self.dst_fmt: str = dst_fmt
        self.sync: int = sync
        self.name: str = "Transpose"

    def __call__(self, imgs: matx.NDArray, apply_index: List[int] = []) -> matx.NDArray:
        assert len(apply_index) == 0, "apply_index is not supported by transpose."
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return self.name + '(from={0}, to={1}, device={2}, sync={3})'.format(
            self.src_fmt, self.dst_fmt, self.device_str, self.sync)
