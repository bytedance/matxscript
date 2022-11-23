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
from .. import ASYNC, PosterizeOp

from ._base import BaseInterfaceClass, BatchRandomBaseClass

class RandomPosterize(BaseInterfaceClass):
    def __init__(self, 
                 bits: int,
                 p: float = 0.5,
                 device_id: int = -2, 
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self._bits:int = bits
        self._p:float = p
    
    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomPosterizeImpl(device, device_str, self._bits, self._p, sync)


class RandomPosterizeImpl(BatchRandomBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 bits: int,
                 p: float = 0.5,
                 sync: int = ASYNC) -> None:
        self.p:float = p
        self.device_str:str = device_str
        super().__init__(prob=self.p)
        self.bits:int = bits
        self.op:PosterizeOp = PosterizeOp(device, bits)
        self.sync:int = sync
        self.name:str = "RandomPosterize"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(imgs, sync=self.sync)

    def __repr__(self)->str:
        return self.name + '(bits={}, p={}, device={}, sync={})'.format(
            self.bits, self.p, self.device_str, self.sync)
