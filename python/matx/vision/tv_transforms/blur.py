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
from .. import ASYNC, GaussianBlurOp

from ._base import BaseInterfaceClass, BatchBaseClass
from ._common import _setup_size


class GaussianBlur(BaseInterfaceClass):
    def __init__(self,
                 kernel_size: List[int],
                 sigma: List[float] = [0.1, 2.0],
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self._kernel_size: Tuple[int, int] = _setup_size(kernel_size)
        for ks in self._kernel_size:
            assert 0 < ks and ks % 2 == 1, "Kernel size value should be an odd and positive number."

        if len(sigma) == 1:
            assert sigma[0] > 0, "If sigma is a single number, it must be positive."
            self._sigma: Tuple[float, float] = (sigma[0], sigma[0])
        elif len(sigma) == 2:
            assert 0. < sigma[0] <= sigma[
                1], "sigma values should be positive and of the form (min, max)."
            self._sigma: Tuple[float, float] = (sigma[0], sigma[1])
        else:
            assert False, "sigma should be a single number or a list/tuple with length 2."

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return GaussianBlurImpl(device, device_str, self._kernel_size, self._sigma, sync)


class GaussianBlurImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float] = (0.1, 2.0),
                 sync: int = ASYNC) -> None:
        super().__init__()

        self.kernel_size: Tuple[int, int] = kernel_size

        self.sigma: Tuple[float, float] = sigma
        self.device_str: str = device_str
        self.sync: int = sync
        self.op: GaussianBlurOp = GaussianBlurOp(device)
        self.name: str = "GaussianBlurImpl"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        size: int = len(imgs)
        ksizes: List[Tuple[int, int]] = [self.kernel_size] * size
        sigmas: List[Tuple[float, float]] = [self.sigma] * size
        return self.op(imgs, ksizes, sigmas, sync=self.sync)

    def __repr__(self) -> str:
        return "{} (kernel_size={},sigma={}, device={}, sync={})".format(
            self.name, self.kernel_size, self.sigma, self.device_str, self.sync)
