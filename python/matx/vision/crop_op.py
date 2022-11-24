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
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _CenterCropOpImpl:
    """ CenterCropOp Impl """

    def __init__(self, device: Any, sizes: Tuple[int, int]) -> None:
        """ Initialize CenterCropOp

        Args:
            device (Any): the matx device used for the operation.
            sizes (Tuple[int, int]): output size for all images, must be 2 dim tuple.
        """
        self.op: matx.NativeObject = make_native_object(
            "VisionCropGeneralOp", device())
        assert len(sizes) == 2, \
            "The sizes len must be equals to 2 in CenterCropOp. "
        self.width: int = sizes[1]
        self.height: int = sizes[0]

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size = len(images)
        x = matx.List()
        x.reserve(batch_size)
        y = matx.List()
        y.reserve(batch_size)
        width = matx.List([self.width] * batch_size)
        height = matx.List([self.height] * batch_size)

        for i in range(batch_size):
            shape_: List[int] = images[i].shape()
            x_: int = (shape_[1] - self.width) // 2
            y_: int = (shape_[0] - self.height) // 2
            x.append(x_)
            y.append(y_)
        return self.op.process(images, x, y, width, height, sync)


class CenterCropOp:
    """ Center crop the given images
    """

    def __init__(self, device: Any, sizes: Tuple[int, int]) -> None:
        """ Initialize CenterCropOp

        Args:
            device (Any): the matx device used for the operation.
            sizes (Tuple[int, int]): output size for all images, must be 2 dim tuple.
        """
        self.op_impl: _CenterCropOpImpl = matx.script(_CenterCropOpImpl)(device=device,
                                                                         sizes=sizes)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ CenterCrop images

        Args:
            images (List[matx.runtime.NDArray]): input images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: center crop images

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import CenterCropOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = CenterCropOp(device=device,
                              size=(224, 224))
        >>> ret = op(nds)
        """
        return self.op_impl(images, sync)


class _CropOpImpl:
    """ CropOp Impl """

    def __init__(self, device: Any) -> None:
        """ Initialize CropOp

        Args:
            device (Any): the matx device used for the operation.
        """
        self.op: matx.NativeObject = make_native_object(
            "VisionCropGeneralOp", device())

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 x: List[int],
                 y: List[int],
                 width: List[int],
                 height: List[int],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        return self.op.process(images, x, y, width, height, sync)


class CropOp:
    """ Crop images in batch on GPU with customized parameters.
    """

    def __init__(self, device: Any) -> None:
        """ Initialize CropOp

        Args:
            device (Any): the matx device used for the operation.
        """
        self.op_impl: _CropOpImpl = matx.script(_CropOpImpl)(device=device)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 x: List[int],
                 y: List[int],
                 width: List[int],
                 height: List[int],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Crop images

        Args:
            images (List[matx.runtime.NDArray]): source/input image
            x (List[int]): the x coordinates of the top_left corner of the cropped region.
            y (List[int]): the y coordinates of the top_left corner of the cropped region.
            width (List[int]): desired width for each cropped image.
            height (List[int]): desired height for each cropped image.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: crop images

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import CropOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> x = [10, 20, 30]
        >>> y = [50, 35, 20]
        >>> widths = [224, 224, 224]
        >>> heights = [224, 224, 224]
        >>> op = CropOp(device=device)
        >>> ret = op(nds, x, y, widths, heights)
        """
        return self.op_impl(images, x, y, width, height, sync)
