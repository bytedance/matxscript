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
from .. import ASYNC, StackOp, TransposeOp
from ._base import BaseInterfaceClass


class ToTensor(BaseInterfaceClass):
    def __init__(self, device_id: int = -2, sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return ToTensorImpl(device, device_str, sync)


class ToTensorImpl:
    def __init__(self,
                 device: Any,
                 device_str: str,
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.stack_op: StackOp = StackOp(device)
        self.transpose_op: TransposeOp = TransposeOp(
            device, input_layout="NHWC", output_layout="NCHW")
        self.device_str: str = device_str
        self.sync: int = sync
        self.name: str = "ToTensor"

    def __call__(self, imgs: List[matx.NDArray], apply_index: List[int] = []) -> matx.NDArray:
        stacked_img = self.stack_op(imgs)
        transposed_img = self.transpose_op(stacked_img, sync=self.sync)
        # res = transposed_img.torch()
        res = transposed_img
        return res

    def __repr__(self) -> str:
        return self.name + '(device={}, sync={})'.format(self.device_str, self.sync)
