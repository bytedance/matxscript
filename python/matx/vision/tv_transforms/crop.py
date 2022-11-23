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

from os import sync
from typing import Tuple, Union, Sequence, Dict, Any, List, overload
import numbers
import random
import sys
matx = sys.modules['matx']
from .. import ASYNC, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101, BORDER_REFLECT
from .. import CropOp, PadWithBorderOp
from ._base import BaseInterfaceClass, BatchBaseClass


class CenterCrop(BaseInterfaceClass):
    def __init__(self,
                 size: List[int],
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        # check size
        if len(size) == 1:
            self._size: Tuple[int, int] = (size[0], size[0])
        elif len(size) == 2:
            self._size: Tuple[int, int] = (size[0], size[1])
        else:
            assert False, "Crop size value should be an integer or a list/tuple with length 2."

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return CenterCropImpl(device, device_str, self._size, sync)


class CenterCropImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 size: Tuple[int, int],
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.size: Tuple[int, int] = size
        self.device_str: str = device_str
        self.sync: int = sync
        fill = (0, 0, 0)
        padding_mode = BORDER_CONSTANT
        self.pad_op: PadWithBorderOp = PadWithBorderOp(device, fill, padding_mode)
        self.crop_op: CropOp = CropOp(device)
        self.name: str = "CenterCropImpl"

    def get_crop_params(self, h: int, w: int) -> Tuple[int, int, int, int]:
        th, tw = self.size
        x = (w - tw) // 2
        y = (h - th) // 2
        return y, x, th, tw

    def get_pad_params(self, h: int, w: int) -> Tuple[int, int, bool]:
        th, tw = self.size
        h_pad, w_pad = 0, 0
        if th > h:
            h_pad = int((1 + th - h) / 2)
        if tw > w:
            w_pad = int((1 + tw - w) / 2)
        need_pad = (h_pad + w_pad) > 0
        return h_pad, w_pad, need_pad

    def _process_crop_op(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        batch_size = len(imgs)
        need_pad = False
        h_pads, w_pads = [], []
        padded_height, padded_width = [], []
        for i in range(batch_size):
            h, w = imgs[i].shape()[:2]
            h_pad, w_pad, need_pad_tmp = self.get_pad_params(h, w)
            need_pad |= need_pad_tmp
            h_pads.append(h_pad)
            w_pads.append(w_pad)
            padded_height.append(h + 2 * h_pad)
            padded_width.append(w + 2 * w_pad)
        if need_pad:
            imgs = self.pad_op(imgs, h_pads, h_pads, w_pads, w_pads, sync=self.sync)
        crop_x, crop_y, crop_w, crop_h = [], [], [], []
        for i in range(batch_size):
            y, x, h, w = self.get_crop_params(padded_height[i], padded_width[i])
            crop_x.append(x)
            crop_y.append(y)
            crop_w.append(w)
            crop_h.append(h)
        imgs = self.crop_op(imgs, crop_x, crop_y, crop_w, crop_h, sync=self.sync)
        return imgs

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self._process_crop_op(imgs)

    def __repr__(self) -> str:
        return self.name + '(size={0}, device={1}, sync={2})'.format(
            self.size, self.device_str, self.sync)


class RandomCrop(BaseInterfaceClass):
    def __init__(self,
                 size: List[int],
                 padding: List[int],
                 pad_if_needed: bool = False,
                 fill: List[int] = [0],
                 padding_mode: str = "constant",
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        # check size
        if len(size) == 1:
            self._size: Tuple[int, int] = (size[0], size[0])
        elif len(size) == 2:
            self._size: Tuple[int, int] = (size[0], size[1])
        else:
            assert False, "Crop size value should be an integer or a list/tuple with length 2."
        # check padding
        if padding is None:
            self._padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
        elif len(padding) == 1:
            self._padding: Tuple[int, int, int, int] = (
                padding[0], padding[0], padding[0], padding[0])
        elif len(padding) == 2:
            self._padding: Tuple[int, int, int, int] = (
                padding[0], padding[1], padding[0], padding[1])
        elif len(padding) == 4:
            self._padding: Tuple[int, int, int, int] = (
                padding[0], padding[1], padding[2], padding[3])
        else:
            assert False, "Padding must be None or a 1, 2 or 4 element tuple.."
        self._pad_if_needed: bool = pad_if_needed
        # check fill
        if len(fill) == 1:
            self._fill: Tuple[int, int, int] = (fill[0], fill[0], fill[0])
        elif len(fill) == 3:
            self._fill: Tuple[int, int, int] = (fill[0], fill[1], fill[2])
        else:
            assert False, "fill value should be a 1 or 3 element tuple."
        # check padding_mode
        assert padding_mode in ["constant", "edge", "reflect",
                                "symmetric"], "padding_mode should be constant, edge, reflect or symmetric."
        self._padding_mode: str = padding_mode

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomCropImpl(
            device,
            device_str,
            self._size,
            self._padding,
            self._pad_if_needed,
            self._fill,
            self._padding_mode,
            sync)


class RandomCropImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 size: Tuple[int, int],
                 padding: Tuple[int, int, int, int],
                 pad_if_needed: bool,
                 fill: Tuple[int, int, int],
                 padding_mode: str,
                 sync: int) -> None:
        super().__init__()
        self.device_str: str = device_str
        self.size: Tuple[int, int] = size
        self.padding: Tuple[int, int, int, int] = padding
        self.pad_if_needed: bool = pad_if_needed
        self.fill: Tuple[int, int, int] = fill
        self.padding_mode: str = padding_mode
        self.sync: int = sync
        self.crop_op: CropOp = CropOp(device)
        torch_padding_mode: Dict[str, str] = {
            "constant": BORDER_CONSTANT,
            "edge": BORDER_REPLICATE,
            "reflect": BORDER_REFLECT_101,
            "symmetric": BORDER_REFLECT
        }
        self.pad_op: PadWithBorderOp = PadWithBorderOp(
            device, self.fill, torch_padding_mode[self.padding_mode])
        self.name: str = "RandomCrop"

    def get_crop_params(self, h: int, w: int) -> Tuple[int, int, int, int]:
        th, tw = self.size
        if w == tw and h == th:
            return 0, 0, h, w
        i = 0
        j = 0
        if h - th > 0:
            i = random.randint(0, h - th)
        else:
            i = random.randint(h - th, 0)
        if w - tw > 0:
            j = random.randint(0, w - tw)
        else:
            j = random.randint(w - tw, 0)
        return i, j, th, tw

    def get_pad_params(self, h: int, w: int) -> Tuple[int, int, int, int, bool]:
        left_p, top_p, right_p, bot_p = self.padding
        if self.pad_if_needed and h + top_p + bot_p < self.size[0]:
            h_pad = self.size[0] - h - top_p - bot_p
            top_p += h_pad
            bot_p += h_pad
        if self.pad_if_needed and w + left_p + right_p < self.size[1]:
            w_pad = self.size[1] - w - left_p - right_p
            left_p += w_pad
            right_p += w_pad
        need_pad = (top_p + bot_p + left_p + right_p) > 0
        return top_p, bot_p, left_p, right_p, need_pad

    def _process_crop_op(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        batch_size = len(imgs)
        need_pad = False
        top_pads, bot_pads, left_pads, right_pads = [], [], [], []
        padded_height, padded_width = [], []
        for i in range(batch_size):
            h, w = imgs[i].shape()[:2]
            top_p, bot_p, left_p, right_p, need_pad_tmp = self.get_pad_params(h, w)
            need_pad |= need_pad_tmp
            top_pads.append(top_p)
            bot_pads.append(bot_p)
            left_pads.append(left_p)
            right_pads.append(right_p)
            padded_height.append(h + top_p + bot_p)
            padded_width.append(w + left_p + right_p)
        if need_pad:
            imgs = self.pad_op(imgs, top_pads, bot_pads, left_pads, right_pads, sync=self.sync)
        crop_x, crop_y, crop_w, crop_h = [], [], [], []
        for i in range(batch_size):
            y, x, h, w = self.get_crop_params(padded_height[i], padded_width[i])
            crop_x.append(x)
            crop_y.append(y)
            crop_w.append(w)
            crop_h.append(h)
        imgs = self.crop_op(imgs, crop_x, crop_y, crop_w, crop_h, sync=self.sync)
        return imgs

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        return self._process_crop_op(imgs)

    def __repr__(self) -> str:
        return self.name + "(size={0}, padding={1}, device={2}, sync={3})".format(
            self.size, self.padding, self.device_str, self.sync)
