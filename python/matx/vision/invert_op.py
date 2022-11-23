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


class _InvertOpImpl:
    """ Impl: Invert all values in images. e.g. turn 20 into 255-20=235
    """

    def __init__(self,
                 device: Any,
                 prob: float = 1.1,
                 per_channel: bool = False,
                 cap_value: float = 255.0) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionLinearAdjustGeneralOp", device())
        self.prob: float = prob
        self.per_channel: bool = per_channel
        self.cap_value: float = cap_value

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        channel_size: int = images[0].shape()[2]
        factor_per_image: int = 1
        if self.per_channel:
            factor_per_image = channel_size

        factors_ = [-1] * (batch_size * factor_per_image)
        shifts_ = [self.cap_value] * (batch_size * factor_per_image)
        if self.prob < 1.0:
            for i in range(batch_size):
                for ch in range(factor_per_image):
                    if random.random() >= self.prob:
                        factors_[i * factor_per_image + ch] = 1.0
                        shifts_[i * factor_per_image + ch] = 0.0

        return self.op.process(
            images,
            factors_,
            shifts_,
            self.per_channel,
            sync)


class InvertOp:
    """ Invert all values in images. e.g. turn 20 into 255-20=235
    """

    def __init__(self,
                 device: Any,
                 prob: float = 1.1,
                 per_channel: bool = False,
                 cap_value: float = 255.0) -> None:
        """ Initialize InvertOp

        Args:
            device (Any) : the matx device used for the operation
            prob (float, optional): probability for inversion. Invert all by default.
            per_channel (float, optional): whether to apply the inversion probability on each image or each channel.
            cap_value (float, optional): the minuend for inversion, 255.0 by default.
        """
        self.op: _InvertOpImpl = matx.script(_InvertOpImpl)(
            device, prob, per_channel, cap_value)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Invert image pixels by substracting itself from given cap value

        Args:
            images (List[matx.runtime.NDArray]): target images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import InvertOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = InvertOp(device)
        >>> ret = op(nds)
        """
        return self.op(images, sync)
