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

from typing import Any, List, Tuple, Union, Dict
from collections.abc import Sequence
import sys
matx = sys.modules['matx']
from .. import ASYNC, CvtColorOp

from ._base import BaseInterfaceClass, BatchBaseClass
from ._common import create_device


class CvtColor(BaseInterfaceClass):
    def __init__(self,
                 color_code: str,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self.op_name: str = "CvtColor"
        self.color_code: str = color_code

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return CvtColorImpl(device, device_str, self.color_code, sync)


class CvtColorImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 color_code: str,
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.device_str: str = device_str
        self.color_code: str = color_code
        self.op: Any = CvtColorOp(device, self.color_code)
        self.sync: int = sync
        self.name: str = "CvtColor"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(imgs, sync=self.sync)

    def __repr__(self) -> str:
        return self.name + '(color_code={}, device={}, sync={})'.format(
            self.color_code, self.device_str, self.sync)
