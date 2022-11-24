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

import math
from typing import List, Any, Tuple
from .constants._sync_mode import ASYNC
from .opencv._cv_border_types import BORDER_CONSTANT
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _PadOpImpl:
    """ PadOp Impl """

    def __init__(self,
                 device: Any,
                 size: Tuple[int, int],
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 pad_type: str = BORDER_CONSTANT,
                 with_corner: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionPadGeneralOp", pad_values, device())
        self.size: Tuple[int, int] = size
        self.dst_height: int = size[0]
        self.dst_width: int = size[1]
        self.pad_type: str = pad_type
        self.with_corner: bool = with_corner

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        image_size = len(images)
        top_pads = matx.List()
        bottom_pads = matx.List()
        left_pads = matx.List()
        right_pads = matx.List()

        top_pads.reserve(image_size)
        bottom_pads.reserve(image_size)
        left_pads.reserve(image_size)
        right_pads.reserve(image_size)

        for index in range(image_size):
            left_pad = 0
            right_pad = 0
            top_pad = 0
            bottom_pad = 0

            shape: List[int] = images[index].shape()
            src_height: int = shape[0]
            src_width: int = shape[1]

            if self.dst_width > src_width:
                # w没对齐, 左右两边pad
                if self.with_corner:
                    right_pad = self.dst_width - src_width
                else:
                    left_pad = (self.dst_width - src_width) // 2
                    right_pad = self.dst_width - src_width - left_pad

            if self.dst_height > src_height:
                # h没对齐, 上下两边pad
                if self.with_corner:
                    bottom_pad = self.dst_height - src_height
                else:
                    top_pad = (self.dst_height - src_height) // 2
                    bottom_pad = self.dst_height - src_height - top_pad

            top_pads.append(top_pad)
            bottom_pads.append(bottom_pad)
            left_pads.append(left_pad)
            right_pads.append(right_pad)

        return self.op.process(
            images,
            top_pads,
            bottom_pads,
            left_pads,
            right_pads,
            self.pad_type,
            sync)


class PadOp:
    """ Forms a border around given image.
    """

    def __init__(self,
                 device: Any,
                 size: Tuple[int, int],
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 pad_type: str = BORDER_CONSTANT,
                 with_corner: bool = False) -> None:
        """ Initialize PadOp

        Args:
            device (Any) : the matx device used for the operation.
            size (Tuple[int, int]): output size for all images, must be 2 dim tuple.
            pad_values (Tuple[int, int, int], optional): Border value if border_type==BORDER_CONSTANT.
                                              Padding value is 3 dim tuple, three channels would be padded with the given value.
                                              Defaults to (0, 0, 0).
            pad_type (str, optional): pad mode, could be chosen from BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, more pad_type see cv_border_types for details.
                                      Defaults to BORDER_CONSTANT.
            with_corner (bool, optional): If True, forms a border in lower right of the image.
                                          Defaults to False.
        """
        self.op_impl: _PadOpImpl = matx.script(_PadOpImpl)(device=device,
                                                           size=size,
                                                           pad_values=pad_values,
                                                           pad_type=pad_type,
                                                           with_corner=with_corner)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Pad input images.

        Args:
            images (List[matx.runtime.NDArray]): input images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: Pad images.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import PadOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = PadOp(device=device,
                       size=(224, 224),
                       pad_values=(0, 0, 0),
                       pad_type=matx.vision.BORDER_CONSTANT)
        >>> ret = op(nds)
        """
        return self.op_impl(images, sync)


class _PadWithBorderOpImpl:
    """ PadWithBorderOp Impl """

    def __init__(self,
                 device: Any,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 pad_type: str = BORDER_CONSTANT) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionPadGeneralOp", pad_values, device())
        self.pad_type: str = pad_type

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 top_pads: List[int],
                 bottom_pads: List[int],
                 left_pads: List[int],
                 right_pads: List[int],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert len(left_pads) == batch_size, "The length of left_pads should be equal to input images size"
        assert len(top_pads) == batch_size, "The length of top_pads should be equal to input images size"
        assert len(right_pads) == batch_size, "The length of right_pads should be equal to input images size"
        assert len(
            bottom_pads) == batch_size, "The length of bottom_pads should be equal to input images size"
        return self.op.process(
            images,
            top_pads,
            bottom_pads,
            left_pads,
            right_pads,
            self.pad_type,
            sync)


class PadWithBorderOp:
    """ Forms a border around given image.
    """

    def __init__(self,
                 device: Any,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 pad_type: str = BORDER_CONSTANT) -> None:
        """ Initialize PadWithBorderOp

        Args:
            device (Any): the matx device used for the operation.
            pad_values (Tuple[int, int, int], optional): Border value if border_type==BORDER_CONSTANT.
                                              Padding value is 3 dim tuple, three channels would be padded with the given value.
                                              Defaults to (0, 0, 0).
            pad_type (str, optional): pad mode, could be chosen from BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, more pad_type see cv_border_types for details.
                                      Defaults to BORDER_CONSTANT.
        """
        self.op_impl: _PadWithBorderOpImpl = matx.script(_PadWithBorderOpImpl)(
            device=device, pad_values=pad_values, pad_type=pad_type)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 top_pads: List[int],
                 bottom_pads: List[int],
                 left_pads: List[int],
                 right_pads: List[int],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Pad input images with border.

        Args:
            images (List[matx.runtime.NDArray]): input images.
            top_pads (List[int]): The number of pixels to pad that above the images.
            bottom_pads (List[int]): The number of pixels to pad that below the images.
            left_pads (List[int]): The number of pixels to pad that to the left of the images.
            right_pads (List[int]): The number of pixels to pad that to the right of the images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: Pad images.

        Example:
        >>> import cv2
        >>> import matx
        >>> from matx.vision import PadWithBorderOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = PadWithBorderOp(device=device,
                                 pad_values=(0, 0, 0),
                                 pad_type=matx.vision.BORDER_CONSTANT)
        >>> ret = op(nds)
        """
        return self.op_impl(images, top_pads, bottom_pads, left_pads, right_pads, sync)
