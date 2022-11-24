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

from typing import Tuple, List, Any, Dict
import random
import sys
matx = sys.modules['matx']
import numpy as np
from .. import ASYNC, SYNC_CPU
from .. import COLOR_RGB2GRAY, COLOR_RGB2HSV_FULL, COLOR_HSV2RGB_FULL
from .. import CvtColorOp, ColorLinearAdjustOp, MixupImagesOp, MeanOp

from ._base import BaseInterfaceClass, BatchBaseClass
from ._common import _assert


class ColorJitter(BaseInterfaceClass):
    def __init__(self,
                 brightness: List[float] = [0],
                 contrast: List[float] = [0],
                 saturation: List[float] = [0],
                 hue: List[float] = [0],
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self.brightness: List[float] = brightness
        self.contrast: List[float] = contrast
        self.saturation: List[float] = saturation
        self.hue: List[float] = hue

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return ColorJitterImpl(
            device,
            device_str,
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
            sync)


class ColorJitterImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 brightness: List[float] = [0],
                 contrast: List[float] = [0],
                 saturation: List[float] = [0],
                 hue: List[float] = [0],
                 sync: int = ASYNC) -> None:
        super().__init__()
        self.device_str: str = device_str
        self.device: Any = device
        self.brightness: List[float] = self._check_input(brightness, "brightness")
        self.contrast: List[float] = self._check_input(contrast, "contrast")
        self.saturation: List[float] = self._check_input(saturation, "saturation")
        self.hue: List[float] = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

        self.rgb_adjust_op: ColorLinearAdjustOp = ColorLinearAdjustOp(self.device)
        self.hsv_adjust_op: ColorLinearAdjustOp = ColorLinearAdjustOp(self.device, per_channel=True)

        self.gray_op: CvtColorOp = CvtColorOp(self.device, COLOR_RGB2GRAY)
        self.hsv_op: CvtColorOp = CvtColorOp(self.device, COLOR_RGB2HSV_FULL)
        self.rgb_op: CvtColorOp = CvtColorOp(self.device, COLOR_HSV2RGB_FULL)

        self.mix_up_op: MixupImagesOp = MixupImagesOp(self.device)
        self.mean_op: MeanOp = MeanOp(self.device)
        self.sync: int = sync
        self.name: str = "ColorJitterImpl"

    def _check_input(self,
                     value: List[float],
                     name: str,
                     center: int = 1,
                     bound: Tuple[float, float] = (0.0, float("inf")),
                     clip_first_on_zero: bool = True) -> List[float]:
        _assert(len(value) > 0 and len(value) <= 2,
                "{} should be a single number or a list with length 2.".format(name))
        if len(value) == 1:
            _assert(
                value[0] >= 0,
                "If {} is a single number, it must be non negative.".format(name))
            value = [center - value[0], center + value[0]]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        _assert(bound[0] <= value[0] <= value[1] <= bound[1],
                "{} values should be between {}".format(name, bound))
        if value[0] == value[1] == center:
            value = []
        return value

    def _brightness(self,
                    imgs: List[matx.NDArray],
                    factors: List[float],
                    sync: int) -> List[matx.NDArray]:
        size: int = len(imgs)
        shifts: List[int] = [0] * size
        return self.rgb_adjust_op(imgs, factors, shifts, sync=sync)

    def _contrast(self,
                  imgs: List[matx.NDArray],
                  factors: List[float],
                  sync: int) -> List[matx.NDArray]:
        gray: List[matx.NDArray] = self.gray_op(imgs)
        mean: matx.NDArray = self.mean_op(gray, sync=SYNC_CPU)
        shifts = []
        for i in range(len(factors)):
            shifts.append(mean[i][0] * (1 - factors[i]))
        imgs = self.rgb_adjust_op(imgs, factors, shifts, sync=sync)
        return imgs

    def _saturation(self,
                    imgs: List[matx.NDArray],
                    factors: List[float],
                    sync: int) -> List[matx.NDArray]:
        gray: List[matx.NDArray] = self.gray_op(imgs)
        factors2: List[float] = [1 - f for f in factors]
        imgs = self.mix_up_op(imgs, gray, factors, factors2, sync=sync)
        return imgs

    def _hue(self, imgs: List[matx.NDArray], factors: List[float], sync: int) -> List[matx.NDArray]:
        size: int = len(imgs)
        hsv: List[matx.NDArray] = self.hsv_op(imgs)  # h [0, 255]
        hsv_shifts: List[float] = []
        hsv_factors: List[float] = []
        for i in range(size):
            hsv_shifts += [factors[i] * 256, 0, 0]
            hsv_factors += [1.0, 1.0, 1.0]
        # TODO we need normal cast here, instead of saturate_cast,
        # update adjust op kernel to allow normal cast
        hsv_adjust: List[matx.NDArray] = self.hsv_adjust_op(hsv, hsv_factors, hsv_shifts)
        imgs = self.rgb_op(hsv_adjust, sync=sync)
        return imgs

    def get_params(self) -> Dict[str, Tuple[int, List[float]]]:
        fn_idx: List[int] = [0, 1, 2, 3]
        # random.shuffle(fn_idx)
        b: List[float] = [] if len(
            self.brightness) == 0 else [
            random.uniform(
                self.brightness[0],
                self.brightness[1])]
        c: List[float] = [] if len(
            self.contrast) == 0 else [
            random.uniform(
                self.contrast[0],
                self.contrast[1])]
        s: List[float] = [] if len(
            self.saturation) == 0 else [
            random.uniform(
                self.saturation[0],
                self.saturation[1])]
        h: List[float] = [] if len(self.hue) == 0 else [random.uniform(self.hue[0], self.hue[1])]
        params: Dict[str, Tuple[int, List[float]]] = {}
        params["brightness"] = (fn_idx[0], b)
        params["contrast"] = (fn_idx[1], c)
        params["saturation"] = (fn_idx[2], s)
        params["hue"] = (fn_idx[3], h)
        return params

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        size: int = len(imgs)
        params: List[Dict[str, Tuple[int, List[float]]]] = [self.get_params() for _ in range(size)]
        rounds_info: List[Dict[str, List[Tuple[int, float]]]] = [{} for _ in range(4)]
        for idx, p in enumerate(params):
            for k, v in p.items():
                if len(v[1]) != 0:
                    if k not in rounds_info[v[0]]:
                        rounds_info[v[0]][k] = []
                    rounds_info[v[0]][k].append((idx, v[1][0]))
        for i, r in enumerate(rounds_info):
            sync: int = ASYNC if i != 3 else self.sync
            for fn_name in r:
                img_idx: List[int] = [item[0] for item in r[fn_name]]
                factors: List[float] = [item[1] for item in r[fn_name]]
                target_imgs: List[matx.NDArray] = self._get_images_from_apply_index(imgs, img_idx)
                result_imgs: List[matx.NDArray] = self._call_func(
                    fn_name, target_imgs, factors, sync)
                imgs = self._put_back_converted_images(imgs, img_idx, result_imgs)
        return imgs

    def _call_func(self,
                   func_name: str,
                   target: List[matx.NDArray],
                   factor: List[float],
                   sync: int) -> List[matx.NDArray]:
        if func_name == "brightness":
            return self._brightness(target, factor, sync)
        if func_name == "contrast":
            return self._contrast(target, factor, sync)
        if func_name == "saturation":
            return self._saturation(target, factor, sync)
        if func_name == "hue":
            return self._hue(target, factor, sync)
        return target

    def __repr__(self) -> str:
        s: str = "ColorJitter(brightness={}, contrast={}, saturation={}, hue={}, device={}, sync={})".format(
            self.brightness, self.contrast, self.saturation, self.hue, self.device_str, self.sync)
        return s
