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
import random

from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _ColorLinearAdjustOpImpl:
    """ ColorLinearAdjustOp Impl """

    def __init__(self,
                 device: Any,
                 prob: float = 1.1,
                 per_channel: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionLinearAdjustGeneralOp", device())
        self.prob: float = prob
        self.per_channel: bool = per_channel

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 factors: List[float],
                 shifts: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        channel_size: int = images[0].shape()[2]
        factor_per_image: int = 1
        assert len(shifts) == len(
            factors), "The length of factors should be equal to the length of shifts"
        if self.per_channel:
            assert len(factors) == channel_size * \
                batch_size, "The length of factors should be equal to input image size times channel size"
            factor_per_image = channel_size
        else:
            assert len(factors) == batch_size, "The length of factors should be equal to input image size"

        if self.prob < 1.0:
            for i in range(batch_size):
                for ch in range(factor_per_image):
                    if random.random() >= self.prob:
                        factors[i * factor_per_image + ch] = 1.0
                        shifts[i * factor_per_image + ch] = 0.0

        return self.op.process(
            images,
            factors,
            shifts,
            self.per_channel,
            sync)


class ColorLinearAdjustOp:
    """ Apply linear adjust on pixels of input images, i.e. apply a * v + b for each pixel v in image/channel.
    """

    def __init__(self,
                 device: Any,
                 prob: float = 1.1,
                 per_channel: bool = False) -> None:
        """ Initialize ColorLinearAdjustOp

        Args:
            device (Any) : the matx device used for the operation
            prob (float, optional) : probability for linear ajustment on each image. Apply on all by default.
            per_channel (bool, optional) : if False, all channels of a single image would use the same linear parameters; if True, each channel would be able to set different linear adjustment
        """
        self.op_impl: _ColorLinearAdjustOpImpl = matx.script(_ColorLinearAdjustOpImpl)(
            device=device, prob=prob, per_channel=per_channel)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 factors: List[float],
                 shifts: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply linear adjust on pixels of input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            factors (List[float]): factor for linear adjustment.
            shifts (List[float]): shift for linear adjustment.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.
        Returns:
            List[matx.runtime.NDArray]: converted images. The output value would be in its original data type range, e.g. for uint [0, 255]

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import ColorLinearAdjustOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> # create parameters for linear adjustment
        >>> factors = [1.1, 1.2, 1.3]
        >>> shifts = [-10, -20, -30]

        >>> op = ColorLinearAdjustOp(device, per_channel=False)
        >>> ret = op(nds, factors, shifts)
        """
        return self.op_impl(images, factors, shifts, sync)
