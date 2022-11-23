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
from .. import ASYNC, StackOp

from ._base import BaseInterfaceClass


class StackOpCpu:
    def __init__(self, device: Any):
        self.device: Any = device

    def __call__(self, imgs: List[matx.NDArray], sync: int = 0) -> matx.NDArray:
        return matx.array.stack(imgs)


class Stack(BaseInterfaceClass):
    def __init__(self,
                 device_id: int = -2,
                 sync: int = ASYNC):
        super().__init__(device_id=device_id, sync=sync)

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return StackImpl(device, device_str, sync)


class StackImpl:
    def __init__(self,
                 device: Any,
                 device_str: str,
                 sync: int = ASYNC) -> None:
        super().__init__()
        if device_str == "cpu":
            self.op: Any = StackOpCpu(device)
        else:
            self.op: Any = StackOp(device)
        self.device_str: str = device_str
        self.sync: int = sync
        self.name: str = "Stack"

    def __call__(self, imgs: List[matx.NDArray], apply_index: List[int] = []) -> matx.NDArray:
        assert len(apply_index) == 0, "apply_index is not supported by stack."
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return self.name + "(device={}, sync={})".format(self.device_str, self.sync)
