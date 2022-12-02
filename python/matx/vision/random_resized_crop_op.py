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
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _RandomResizedCropOpImpl:
    """ RandomResizedCropOp Impl """

    def __init__(self,
                 device: Any,
                 size: Tuple[int, int],
                 scale: List[float],
                 ratio: List[float],
                 interp: str = INTER_LINEAR) -> None:
        self.interp: str = interp
        self.des_width: int = size[1]
        self.des_height: int = size[0]
        self.op: matx.NativeObject = make_native_object(
            "VisionRandomResizedCropGeneralOp", scale, ratio, device())

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size = len(images)
        des_widths = [self.des_width] * batch_size
        des_heights = [self.des_height] * batch_size
        return self.op.process(images, des_heights, des_widths, self.interp, sync)


class RandomResizedCropOp:
    """ RandomResizedCropOp given image on gpu.
    """

    def __init__(self,
                 device: Any,
                 size: Tuple[int, int],
                 scale: List[float],
                 ratio: List[float],
                 interp: str = INTER_LINEAR) -> None:
        """ Initialize RandomResizedCropOP

        Args:
            device (Any): the matx device used for the operation.
            size (Tuple[int, int]): output size for all images, must be 2 dim tuple.
            scale (List[float]): Specifies the lower and upper bounds for the random area of
                           the crop, before resizing. The scale is defined with respect
                           to the area of the original image.
            ratio (List[float]): lower and upper bounds for the random aspect ratio of the crop,
                           before resizing.
            interp (str, optional): Desired interpolation.
                                    INTER_NEAREST -- a nearest-neighbor interpolation;
                                    INTER_LINEAR -- a bilinear interpolation (used by default);
                                    INTER_CUBIC -- a bicubic interpolation over 4x4 pixel neighborhood;
                                    PILLOW_INTER_LINEAR  -- a bilinear interpolation, simalir to Pillow(only support GPU)
                                    Defaults to INTER_LINEAR.
        """
        self.op_impl: _RandomResizedCropOpImpl = matx.script(_RandomResizedCropOpImpl)(
            device=device, size=size, scale=scale, ratio=ratio, interp=interp)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Resize and Crop image depends on scale and ratio.

        Args:
            images (List[matx.runtime.NDArray]): input images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: RandomResizedCrop images.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import RandomResizedCropOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = RandomResizedCropOp(device=device,
                                     size=(224, 224),
                                     scale=[0.8, 1.0],
                                     ratio=[0.8, 1.25],
                                     interp=matx.vision.INTER_LINEAR)
        >>> ret = op(nds)
        """
        return self.op_impl(images, sync)
