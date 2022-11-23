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

from typing import Any, Dict, Tuple, List
import sys
matx = sys.modules['matx']
from .. import ASYNC, RotateOp, WarpAffineOp, WarpPerspectiveOp
import math

from ._base import BaseInterfaceClass, BatchRandomBaseClass, BatchBaseClass
from ._common import _torch_interp_mode, _setup_angle, _check_sequence_input, _uniform_random, _randint


class RandomRotation(BaseInterfaceClass):
    def __init__(self,
                 degrees: List[float],
                 interpolation: str = 'nearest',
                 expand: bool = False,
                 center: List[int] = [],
                 fill: List[float] = [],
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self.degrees: List[float] = _setup_angle(degrees, name="degrees", req_sizes=[2])

        assert interpolation in _torch_interp_mode
        self.interp: str = _torch_interp_mode[interpolation]

        if len(fill) == 0:
            self.fill: Tuple[float, float, float] = (0, 0, 0)
        elif len(fill) == 1:
            self.fill: Tuple[float, float, float] = (fill[0], fill[0], fill[0])
        else:
            self.fill: Tuple[float, float, float] = (fill[0], fill[1], fill[2])

        if len(center) != 0:
            _check_sequence_input(center, "center", req_sizes=[2])
        self.center: List[int] = center
        self.expand: bool = expand

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomRotationImpl(device,
                                  device_str,
                                  self.degrees,
                                  self.interp,
                                  self.expand,
                                  self.center,
                                  self.fill,
                                  sync)


class RandomRotationImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 degrees: List[float],
                 interpolation: str,
                 expand: bool,
                 center: List[int],
                 fill: Tuple[float, float, float],
                 sync: int = ASYNC) -> None:
        self.device_str: str = device_str
        super().__init__()
        self.degrees: List[float] = degrees
        self.interpolation: str = interpolation
        self.expand: bool = expand
        self.center: List[int] = center
        self.fill: Tuple[float, float, float] = fill
        self.op: RotateOp = RotateOp(device,
                                     pad_values=self.fill,
                                     interp=self.interpolation,
                                     expand=self.expand)
        self.sync: int = sync
        self.name: str = "RandomRotation"

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        batch_size: int = len(imgs)
        angles: List[float] = _uniform_random(self.degrees[0], self.degrees[1], batch_size)
        center: List[Tuple[int, int]] = []
        if len(self.center) == 2:
            center = [(self.center[0], self.center[1]) for _ in range(batch_size)]
        imgs = self.op(imgs, angles, center, self.sync)
        return imgs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(self.interpolation)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ", device={0}, sync={1}".format(self.device_str, self.sync)
        format_string += ')'
        return format_string


class RandomAffine(BaseInterfaceClass):
    def __init__(self,
                 degrees: List[float],
                 translate: List[float] = [],
                 scale: List[float] = [],
                 shear: List[float] = [],
                 interpolation: str = 'nearest',
                 fill: List[float] = [],
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)

        assert interpolation in _torch_interp_mode
        self.interp: str = _torch_interp_mode[interpolation]

        self.degrees: List[float] = _setup_angle(degrees, name="degrees", req_sizes=[2])

        if len(translate) != 0:
            _check_sequence_input(translate, "translate", req_sizes=[2])
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate: List[float] = translate

        if len(scale) != 0:
            _check_sequence_input(scale, "scale", req_sizes=[2])
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale: List[float] = scale

        if len(shear) != 0:
            self.shear: List[float] = _setup_angle(shear, name="shear", req_sizes=[2, 4])
        else:
            self.shear: List[float] = shear

        if len(fill) == 0:
            self.fill: Tuple[float, float, float] = (0, 0, 0)
        elif len(fill) == 1:
            self.fill: Tuple[float, float, float] = (fill[0], fill[0], fill[0])
        else:
            self.fill: Tuple[float, float, float] = (fill[0], fill[1], fill[2])

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomAffineImpl(
            device,
            device_str,
            self.degrees,
            self.translate,
            self.scale,
            self.shear,
            self.interp,
            self.fill,
            sync)


class RandomAffineImpl(BatchBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 degrees: List[float],
                 translate: List[float] = [],
                 scale: List[float] = [],
                 shear: List[float] = [],
                 interpolation: str = 'nearest',
                 fill: Tuple[float, float, float] = (0, 0, 0),
                 sync: int = ASYNC) -> None:
        self.device_str: str = device_str
        super().__init__()
        self.degrees: List[float] = degrees
        self.interpolation: str = interpolation
        self.fill: Tuple[float, float, float] = fill
        self.translate: List[float] = translate
        self.scale: List[float] = scale
        self.shear: List[float] = shear
        self.op: RotateOp = WarpAffineOp(device, pad_values=self.fill, interp=self.interpolation)
        self.sync: int = sync
        self.name: str = "RandomAffine"

    def _get_matrix(self, img_shape: List[int]) -> List[List[float]]:
        h: int = img_shape[0]
        w: int = img_shape[1]

        angle: float = _uniform_random(self.degrees[0], self.degrees[1], 1)[0]

        if len(self.translate) != 0:
            max_dx: float = float(w * self.translate[0])
            max_dy: float = float(h * self.translate[1])
            tx: float = _uniform_random(-max_dx, max_dx, 1)[0]
            ty: float = _uniform_random(-max_dy, max_dy, 1)[0]
            translations: Tuple[float, float] = (tx, ty)
        else:
            translations: Tuple[float, float] = (0, 0)

        if len(self.scale) != 0:
            scale: float = _uniform_random(self.scale[0], self.scale[1], 1)[0]
        else:
            scale: float = 1.0

        shear_x: float = 0.0
        shear_y: float = 0.0
        if len(self.shear) != 0:
            shear_x = _uniform_random(self.shear[0], self.shear[1], 1)[0]
            if len(self.shear) == 4:
                shear_y = _uniform_random(self.shear[2], self.shear[3], 1)[0]
        shear: Tuple[float, float] = (shear_x, shear_y)
        center: Tuple[float, float] = (w * 0.5, h * 0.5)
        return self._create_matrix_with_param(center, angle, translations, scale, shear)

    def _create_matrix_with_param(self,
                                  center: Tuple[float,
                                                float],
                                  angle: float,
                                  translate: Tuple[float,
                                                   float],
                                  scale: float,
                                  shear: Tuple[float,
                                               float]) -> List[List[float]]:
        rot: float = math.radians(angle)
        sx: float = math.radians(shear[0])
        sy: float = math.radians(shear[1])

        cx: float = center[0]
        cy: float = center[1]
        tx: float = translate[0]
        ty: float = translate[1]

        # RSS without scaling
        a: float = math.cos(rot - sy) / math.cos(sy)
        b: float = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c: float = math.sin(rot - sy) / math.cos(sy)
        d: float = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        matrix: List[List[float]] = [[a * scale, b * scale, 0], [c * scale, d * scale, 0]]
        matrix[0][2] = -cx * matrix[0][0] - cy * matrix[0][1] + cx + tx
        matrix[1][2] = -cx * matrix[1][0] - cy * matrix[1][1] + cy + ty
        return matrix

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        matrix: List[List[List[float]]] = [self._get_matrix(i.shape()) for i in imgs]
        imgs = self.op(imgs, matrix, sync=self.sync)
        return imgs

    def __repr__(self) -> str:
        s: str = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        s += ', interpolation={interp}'
        if self.fill != 0:
            s += ', fill={fill}'
        s += ", device={0}, sync={1}".format(self.device_str, self.sync)
        s += ')'
        d: Dict[str, Any] = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


class RandomPerspective(BaseInterfaceClass):
    def __init__(self,
                 distortion_scale: float = 0.5,
                 interpolation: str = "bilinear",
                 fill: List[float] = [],
                 p: float = 0.5,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        super().__init__(device_id=device_id, sync=sync)
        self.p: float = p
        assert interpolation in _torch_interp_mode
        self.interp: str = _torch_interp_mode[interpolation]

        self.distortion_scale: float = distortion_scale

        if len(fill) == 0:
            self.fill: Tuple[float, float, float] = (0, 0, 0)
        elif len(fill) == 1:
            self.fill: Tuple[float, float, float] = (fill[0], fill[0], fill[0])
        else:
            self.fill: Tuple[float, float, float] = (fill[0], fill[1], fill[2])
        self.sync: int = sync

    def __call__(self, device: Any, device_str: str, sync: int) -> Any:
        return RandomPerspectiveImpl(
            device,
            device_str,
            self.distortion_scale,
            self.interp,
            self.fill,
            self.p,
            sync)


class RandomPerspectiveImpl(BatchRandomBaseClass):
    def __init__(self,
                 device: Any,
                 device_str: str,
                 distortion_scale: float = 0.5,
                 interpolation: str = "bilinear",
                 fill: Tuple[float, float, float] = (0, 0, 0),
                 p: float = 0.5,
                 sync: int = ASYNC) -> None:
        self.p: float = p
        self.device_str: str = device_str
        super().__init__(p)
        self.interpolation: str = interpolation
        self.distortion_scale: float = distortion_scale
        self.fill: Tuple[float, float, float] = fill
        self.op: RotateOp = WarpPerspectiveOp(
            device, pad_values=self.fill, interp=self.interpolation)
        self.sync: int = sync
        self.name: str = "RandomPerspective"

    def get_points(self, image_shape: List[int]) -> List[List[Tuple[float, float]]]:
        h: int = image_shape[0]
        w: int = image_shape[1]
        half_h: int = h // 2
        half_w: int = w // 2
        dh_w: int = int(self.distortion_scale * half_w) + 1
        dh_h: int = int(self.distortion_scale * half_h) + 1
        topleft: Tuple[float, float] = (
            float(_randint(0, dh_w, 1)[0]),
            float(_randint(0, dh_h, 1)[0])
        )
        topright: Tuple[float, float] = (
            float(_randint(w - dh_w, w, 1)[0]),
            float(_randint(0, dh_h, 1)[0])
        )
        botright: Tuple[float, float] = (
            float(_randint(w - dh_w, w, 1)[0]),
            float(_randint(h - dh_h, h, 1)[0])
        )
        botleft: Tuple[float, float] = (
            float(_randint(0, dh_w, 1)[0]),
            float(_randint(h - dh_h, h, 1)[0])
        )
        startpoints: List[Tuple[float, float]] = [
            (0.0, 0.0), (float(w - 1), 0.0), (float(w - 1), float(h - 1)), (0.0, float(h - 1))]
        endpoints: List[Tuple[float, float]] = [topleft, topright, botright, botleft]
        return [startpoints, endpoints]

    def _process(self, imgs: List[matx.NDArray]) -> List[matx.NDArray]:
        points: List[List[List[Tuple[float, float]]]] = [self.get_points(i.shape()) for i in imgs]
        imgs = self.op(imgs, points, sync=self.sync)
        return imgs

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(p={}, device={}, sync={})'.format(
            self.p, self.device_str, self.sync)
