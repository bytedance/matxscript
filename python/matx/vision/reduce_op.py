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

from typing import Any, List
from .constants._sync_mode import ASYNC
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _SumOpImpl:
    """ Sum Impl """

    def __init__(self, device: Any, per_channel: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionSumOrMeanGeneralOp", device())
        self.per_channel: bool = per_channel

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        return self.op.process(images, self.per_channel, False, sync)


class SumOp:
    """ Sum over each image.
    """

    def __init__(self, device: Any, per_channel: bool = False) -> None:
        """ Initialize SumOp

        Args:
            device (Any) : the matx device used for the operation.
            per_channel (bool, optional) : if True, sum over each channel; if False, sum over the whole image.
        """
        self.op: _SumOpImpl = matx.script(_SumOpImpl)(device, per_channel)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        """ Sum over each image.

        Args:
            images (List[matx.runtime.NDArray]) : target images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.
        Returns:
            matx.runtime.NDArray: summation result. For N images, the result would be shape Nx1 if per_channel is False, otherwise NxC where C is the image channel size.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import SumOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = SumOp(device, per_channel = False)
        >>> ret = op(nds)
        """
        return self.op(images, sync)


class _MeanOpImpl:
    """ Mean Impl """

    def __init__(self, device: Any, per_channel: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionSumOrMeanGeneralOp", device())
        self.per_channel: bool = per_channel

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        return self.op.process(images, self.per_channel, True, sync)


class MeanOp:
    """ Calculate mean over each image.
    """

    def __init__(self, device: Any, per_channel: bool = False) -> None:
        """ Initialize MeanOp

        Args:
            device (Any) : the matx device used for the operation.
            per_channel (bool, optional) : if True, calculate mean over each channel; if False, calculate mean over the whole image.
        """
        self.op: _MeanOpImpl = matx.script(_MeanOpImpl)(device, per_channel)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        """ Calculate mean over each image.

        Args:
            images (List[matx.runtime.NDArray]) : target images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.
        Returns:
            matx.runtime.NDArray: mean result. For N images, the result would be shape Nx1 if per_channel is False, otherwise NxC where C is the image channel size.

        Example:
        >>> import cv2
        >>> import matx
        >>> from matx.vision import MeanOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = MeanOp(device, per_channel = False)
        >>> ret = op(nds)
        """
        return self.op(images, sync)
