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

from typing import Any, List, Dict

from .flip import RandomHorizontalFlip, RandomHorizontalFlipImpl, RandomVerticalFlip, RandomVerticalFlipImpl
from .blur import GaussianBlur, GaussianBlurImpl
from .color_jitter import ColorJitter, ColorJitterImpl
from .contrast import RandomAutocontrast, RandomAutocontrastImpl
from .convert_dtype import ConvertImageDtype, ConvertImageDtypeImpl
from .decode import Decode, DecodeImpl
from .equalize import RandomEqualize, RandomEqualizeImpl
from .invert import RandomInvert, RandomInvertImpl
from .normalize import Normalize, NormalizeImpl
from .posterize import RandomPosterize, RandomPosterizeImpl
from .sharp import RandomAdjustSharpness, RandomAdjustSharpnessImpl
from .solarize import RandomSolarize, RandomSolarizeImpl
from .stack import Stack, StackImpl
from .transpose import Transpose, TransposeImpl
from .to_tensor import ToTensor, ToTensorImpl
from .crop import CenterCrop, CenterCropImpl, RandomCrop, RandomCropImpl
from .grayscale import RandomGrayscale, Grayscale
from .pad import Pad, PadImpl
from .crop import CenterCrop, CenterCropImpl, RandomCrop, RandomCropImpl
from .cvt_color import CvtColor, CvtColorImpl
from .resize import Resize, ResizeImpl, RandomResizedCrop, RandomResizedCropImpl
from .warp import RandomRotation, RandomRotationImpl, RandomAffine, RandomAffineImpl, RandomPerspective, RandomPerspectiveImpl

from .. import ASYNC, SYNC
import torch
from ._common import DeviceManager

__all__ = [
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "GaussianBlur",
    "ColorJitter",
    "RandomAutocontrast",
    "ConvertImageDtype",
    "Decode",
    "RandomEqualize",
    "RandomInvert",
    "Normalize",
    "RandomPosterize",
    "RandomAdjustSharpness",
    "RandomSolarize",
    "Stack",
    "Transpose",
    "ToTensor",
    "CenterCrop",
    "RandomCrop",
    "RandomGrayscale",
    "Grayscale",
    "Pad",
    "CenterCrop",
    "RandomCrop",
    "CvtColor",
    "Resize",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomAffine",
    "RandomPerspective",
    "DeviceManager",
    "Compose",
    "set_device"
]


class Compose(object):
    def __init__(self, device_id: int, transforms: List[Any]) -> None:
        self.default_device_id: int = device_id
        self.transforms: List[Any] = []
        self.device_str: Dict[int, str] = {}
        set_last_op_sync: bool = False
        op_len: int = len(transforms)
        for i in range(op_len):
            op: Any = transforms[op_len - i - 1]
            op_device_id: int = op.device_id()
            if op_device_id == -2:
                op_device_id = self.default_device_id
            op_device: Any = DeviceManager(op_device_id)
            op_device_str: str = self._create_device_str(op_device_id)
            op_has_sync: bool = op.has_sync()
            op_sync: int = ASYNC
            if op_has_sync:
                if not set_last_op_sync:
                    op_sync = SYNC
                    set_last_op_sync = True
                else:
                    op_sync = op.sync()
            self.transforms = [op(op_device, op_device_str, op_sync)] + self.transforms

    def _create_device_str(self, device_id: int) -> str:
        if device_id in self.device_str:
            return self.device_str[device_id]
        cur_device_str: str = "cpu"
        if device_id >= 0:
            cur_device_str = "gpu:{}".format(device_id)
        self.device_str[device_id] = cur_device_str
        return cur_device_str

    def __call__(self, imgs: Any) -> Any:
        for t in self.transforms:
            imgs = t(imgs, [])
        return imgs

    def __repr__(self) -> str:
        format_string = "Compose" + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class set_device():
    def __init__(self, device_id):
        self.device_id = device_id
        self.ori_device_id = None

    def __enter__(self):
        self.ori_device_id = torch.cuda.current_device()
        if self.device_id > 0:
            torch.cuda.set_device(self.device_id)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.ori_device_id:
            torch.cuda.set_device(self.ori_device_id)
