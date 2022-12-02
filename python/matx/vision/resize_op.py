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

from typing import List, Any, Tuple
from .constants._sync_mode import ASYNC
from .opencv._cv_interpolation_flags import INTER_LINEAR
from .constants._resize_mode import *
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _ResizeOpImpl:
    """ ResizeOp Impl """

    def __init__(self,
                 device: Any,
                 size: Tuple[int, int] = (-1, -1),
                 max_size: int = 0,
                 interp: str = INTER_LINEAR,
                 mode: str = RESIZE_DEFAULT) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionResizeGeneralOp", device())
        if len(size) != 2:
            assert False, "The target size for resize op should be 2."
        self.height: int = size[0]
        self.width: int = size[1]
        self.max_size: int = max_size
        self.interp: str = interp
        self.mode: str = mode
        self.use_unique_size: bool = self.height > 0

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 size: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        use_unique_size: bool = self.use_unique_size
        if len(size) > 0:
            use_unique_size = False
            if len(size) != batch_size:
                assert False, "The length of size should be equal to input images"
        else:
            # size is not defined neither in op init, nor in functional call
            # throw exception
            assert use_unique_size, "The target size should be defined either in op initialization, or in runtime"

        desired_height = matx.List()
        desired_width = matx.List()
        desired_height.reserve(batch_size)
        desired_width.reserve(batch_size)
        for i in range(batch_size):
            cur_height: int = self.height
            cur_width: int = self.width
            if not use_unique_size:
                cur_size: Tuple[int, int] = size[i]
                if len(cur_size) != 2:
                    assert False, "The target size for each image should be 2."
                cur_height = cur_size[0]
                cur_width = cur_size[1]

            cur_image: matx.runtime.NDArray = images[i]
            img_height: int = cur_image.shape()[0]
            img_width: int = cur_image.shape()[1]
            cur_ratio: float = img_width / img_height
            width_scale: float = cur_width / img_width
            height_scale: float = cur_height / img_height

            if self.mode == RESIZE_NOT_LARGER:
                if width_scale < height_scale:
                    cur_height = int(img_height * width_scale)
                elif width_scale > height_scale:
                    cur_width = int(img_width * height_scale)

            elif self.mode == RESIZE_NOT_SMALLER:
                if width_scale > height_scale:
                    cur_height = int(img_height * width_scale)
                elif width_scale < height_scale:
                    cur_width = int(img_width * height_scale)

                if cur_ratio > 1.0 and 0 < self.max_size < cur_width:
                    cur_width = self.max_size
                    cur_height = int(self.max_size / cur_ratio)
                elif cur_ratio < 1.0 and 0 < self.max_size < cur_height:
                    cur_height = self.max_size
                    cur_width = int(self.max_size * cur_ratio)
            # fix image_width or image_height is zero
            if cur_height == 0:
                cur_height = 1
            if cur_width == 0:
                cur_width = 1
            desired_height.append(cur_height)
            desired_width.append(cur_width)

        return self.op.process(images, desired_height, desired_width, self.interp, sync)


class ResizeOp:
    """ Resize input images.
    """

    def __init__(self,
                 device: Any,
                 size: Tuple[int, int] = (-1, -1),
                 max_size: int = 0,
                 interp: str = INTER_LINEAR,
                 mode: str = RESIZE_DEFAULT) -> None:
        """ Initialize ResizeOp

        Args:
            device (Any) : the matx device used for the operation.
            size (Tuple[int, int], optional) : output size for all images, must be 2 dim tuple. If omitted, the size must be given when calling.
            max_size (int, optional) : used in RESIZE_NOT_SMALLER mode to make sure output size is not too large.
            interp (str, optional) : desired interpolation method.
                                     INTER_NEAREST -- a nearest-neighbor interpolation;
                                     INTER_LINEAR -- a bilinear interpolation (used by default);
                                     INTER_CUBIC -- a bicubic interpolation over 4x4 pixel neighborhood;
                                     PILLOW_INTER_LINEAR  -- a bilinear interpolation, simalir to Pillow(only support GPU)
                                     INTER_LINEAR by default.
            mode (str, optional) : resize mode, could be chosen from RESIZE_DEFAULT, RESIZE_NOT_LARGER, and RESIZE_NOT_SMALLER
                                   RESIZE_DEFAULT -- resize to the target output size
                                   RESIZE_NOT_LARGER -- keep the width/height ratio, final output size would be one dim equal to target, one dim smaller. e.g. original image shape (360, 240), target size (480, 360), output size (480, 320)
                                   RESIZE_NOT_SMALLER -- keep the width/height ratio, final output size would be one dim equal to target, one dim larger. e.g. original image shape (360, 240), target size (480, 360), output size (540, 360)
                                   RESIZE_DEFAULT by default.
        """
        self.op_impl: _ResizeOpImpl = matx.script(_ResizeOpImpl)(device=device,
                                                                 size=size,
                                                                 max_size=max_size,
                                                                 interp=interp,
                                                                 mode=mode)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 size: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Resize input images.

        Args:
            images (List[matx.runtime.NDArray]) : target images.
            size (List[Tuple[int, int]], optional) : target size for each image, must be 2 dim tuple (h, w).
                                           If omitted, the target size set in op initialization would be used for all images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import ResizeOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = ResizeOp(device, size=(224, 224), mode=matx.vision.RESIZE_NOT_SMALLER)
        >>> ret = op(nds)
        """
        return self.op_impl(images, size, sync)
