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

from typing import Tuple, Union, Sequence, Dict, Any, List
import sys
matx = sys.modules['matx']
from .. import ASYNC, RESIZE_NOT_SMALLER, RESIZE_DEFAULT
from .. import INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
from .. import ResizeOp, RandomResizedCropOp
from ._base import BaseInterfaceClass, BatchBaseClass
from ._common import _setup_size, _torch_interp_mode, create_device


class Resize(BaseInterfaceClass):
    def __init__(self,
                 size: List[int],
                 interpolation: str = "bilinear",
                 resize_mode: str = "",
                 max_size: int = 0,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        # check size
        if len(size) == 1:
            self._size: Tuple[int, int] = (size[0], size[0])
            if not resize_mode:
                self._resize_mode: str = RESIZE_NOT_SMALLER
            else:
                self._resize_mode: str = resize_mode
        elif len(size) == 2:
            self._size: Tuple[int, int] = (size[0], size[1])
            if not resize_mode:
                self._resize_mode: str = RESIZE_DEFAULT
            else:
                self._resize_mode: str = resize_mode
        else:
            assert False, "Resize size value should be an integer or a list/tuple with length 2."
        self._max_size: int = max_size
        # check interpolation mode
        assert interpolation in ["nearest", "bilinear", "bicubic",
                                 "lanczos"], "interpolation_mode should be nearest, bilinear, bicubic or lanczos."
        torch_interp_mode: Dict[str, str] = {
            "nearest": INTER_NEAREST,
            "bilinear": INTER_LINEAR,
            "bicubic": INTER_CUBIC,
            "lanczos": INTER_LANCZOS4
        }
        self._interpolation_mode: str = torch_interp_mode[interpolation]

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return ResizeImpl(
            device,
            device_str,
            self._size,
            self._max_size,
            self._resize_mode,
            self._interpolation_mode,
            sync)


class ResizeImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 size: Tuple[int, int],
                 max_size: int,
                 resize_mode: str,
                 interpolation_mode: str,
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.device_str: str = device_str
        self.size: Tuple[int, int] = size
        self.max_size: int = max_size
        self.resize_mode: str = resize_mode
        self.interpolation_mode: str = interpolation_mode
        self.sync: int = sync
        self.op: ResizeOp = ResizeOp(
            device,
            self.size,
            self.max_size,
            self.interpolation_mode,
            self.resize_mode)
        self.name: str = "ResizeImpl"

    def _process_resize_op(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        imgs = self.op(imgs, sync=self.sync)
        return imgs

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self._process_resize_op(imgs)

    def __repr__(self) -> str:
        return self.name + '(size={0}, interpolation={1}, max_size={2}, device={3}, sync={4})'.format(
            self.size, self.interpolation_mode, self.max_size, self.device_str, self.sync)


class RandomResizedCrop(BaseInterfaceClass):
    def __init__(self,
                 size: List[int],
                 scale: Tuple[float, float] = (0.08, 1.0),
                 ratio: Tuple[float, float] = (3. / 4., 4. / 3.),
                 interpolation: str = "bilinear",
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        # check size
        if len(size) == 1:
            self._size: Tuple[int, int] = (size[0], size[0])
        elif len(size) == 2:
            self._size: Tuple[int, int] = (size[0], size[1])
        else:
            assert False, "Resize size value should be an integer or a list/tuple with length 2."
        # check scale
        if len(scale) == 2:
            assert scale[0] < scale[1], "Scale should be of kind [min, max]."
            self._scale: List[float] = [scale[0], scale[1]]
        else:
            assert False, "Scale should be a tuple with length 2."
        # check ratio
        if len(ratio) == 2:
            assert ratio[0] < ratio[1], "Ratio should be of kind [min, max]."
            self._ratio: List[float] = [ratio[0], ratio[1]]
        else:
            assert False, "Ratio should be a tuple with length 2."
        # check interpolation mode
        assert interpolation in ["nearest", "bilinear", "bicubic",
                                 "lanczos"], "interpolation_mode should be nearest, bilinear, bicubic or lanczos."
        torch_interp_mode: Dict[str, str] = {
            "nearest": INTER_NEAREST,
            "bilinear": INTER_LINEAR,
            "bicubic": INTER_CUBIC,
            "lanczos": INTER_LANCZOS4
        }
        self._interpolation_mode: str = torch_interp_mode[interpolation]

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomResizedCropImpl(
            device,
            device_str,
            self._size,
            self._scale,
            self._ratio,
            self._interpolation_mode,
            sync)


class RandomResizedCropImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 size: Tuple[int, int],
                 scale: List[float],
                 ratio: List[float],
                 interpolation_mode: str,
                 sync: int) -> None:
        self.device_str: str = device_str
        self.size: Tuple[int, int] = size
        self.scale: List[float] = scale
        self.ratio: List[float] = ratio
        self.interpolation_mode: str = interpolation_mode
        self.sync: int = sync
        self.op: RandomResizedCropOp = RandomResizedCropOp(
            device, self.size, self.scale, self.ratio, self.interpolation_mode)
        self.name: str = "RandomResizedCropImpl"

    def _process_random_resized_crop_op(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        imgs = self.op(imgs, sync=self.sync)
        return imgs

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self._process_random_resized_crop_op(imgs)

    def __repr__(self) -> str:
        return self.name + '(size={0}, scale={1}, ratio={2}, interpolation={3}, device={4}, sync={5})'.format(
            self.size, self.scale, self.ratio, self.interpolation_mode, self.device_str, self.sync)
