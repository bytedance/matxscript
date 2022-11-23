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

from typing import Any, List, Tuple
from .constants._sync_mode import ASYNC
from .opencv._cv_border_types import BORDER_CONSTANT
from .opencv._cv_interpolation_flags import INTER_LINEAR
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _RotateOpImpl:
    """ Impl: Apply image rotation.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_CONSTANT,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 interp: str = INTER_LINEAR,
                 expand: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionRotateGeneralOp", device())
        self.pad_type: str = pad_type
        self.pad_values: Tuple[int, int, int] = pad_values
        self.interp: str = interp
        self.expand: bool = expand
        self.scale: float = 1.0

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 angles: List[float],
                 center: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert len(
            angles) == batch_size, "The angle size for rotate op should be equal to batch size."
        if len(center) != 0 and len(center) != batch_size:
            assert False, "The center for rotate op should either be empty (which will use the original image center as the rotate center), or its size sould be equal to batch size"

        expand_ = matx.List()
        expand_.reserve(batch_size)
        scale_ = matx.List()
        scale_.reserve(batch_size)
        dsize_ = matx.List()
        dsize_.reserve(batch_size)
        center_ = matx.List()
        center_.reserve(batch_size)
        use_image_center: bool = len(center) == 0

        for i in range(batch_size):
            image: matx.runtime.NDArray = images[i]
            height: int = image.shape()[0]
            width: int = image.shape()[1]
            dsize_.append([height, width])
            expand_.append(self.expand)
            scale_.append(self.scale)
        if len(center) == 0:
            for i in range(batch_size):
                center_.append([dsize_[i][0] // 2, dsize_[i][1] // 2])
        else:
            for i in range(batch_size):
                assert len(
                    center[i]) == 2, "The rotate center for each image should contain exactly 2 elements (x, y)"
                center_.append([center[i][0], center[i][1]])

        return self.op.process(
            images,
            dsize_,
            center_,
            angles,
            scale_,
            expand_,
            self.pad_type,
            self.pad_values,
            self.interp,
            sync)


class RotateOp:
    """ Apply image rotation.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_CONSTANT,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 interp: str = INTER_LINEAR,
                 expand: bool = False) -> None:
        """ Initialize RotateOp

        Args:
            device (Any) : the matx device used for the operation
            pad_type (str, optional) : border type to fill the target image, use constant value by default.
            pad_values (Tuple[int, int, int], optional) : the border value to fill the target image if pad_type is BORDER_CONSTANT, (0, 0, 0) by default.
            interp (str, optional) : desired interpolation method.
                                     INTER_NEAREST -- a nearest-neighbor interpolation;
                                     INTER_LINEAR -- a bilinear interpolation (used by default);
                                     INTER_CUBIC -- a bicubic interpolation over 4x4 pixel neighborhood;
                                     INTER_LINEAR by default.
            expand (bool, optional) : control the shape of rotated image. If False, the rotated images would be center cropped into the original size;
                                      if True, expand the output to make it large enough to hold the entire rotated image.
        """
        self.op: _RotateOpImpl = matx.script(_RotateOpImpl)(
            device, pad_type, pad_values, interp, expand)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 angles: List[float],
                 center: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply rotation on images.

        Args:
            images (List[matx.runtime.NDArray]) : target images.
            angles (List[float]) : rotation angle for each image
            center (List[Tuple[int, int]], optional) : rotation center (y, x) for each image, if omitted, the image center would be used as rotation center.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import RotateOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> angles = [10, 20, 30]

        >>> op = RotateOp(device, expand = True)
        >>> ret = op(nds, angles)
        """
        return self.op(images, angles, center, sync)
