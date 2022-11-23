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

from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _GammaContrastOpImpl:
    """ GammaContrast Impl """

    def __init__(self, device: Any, per_channel: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionGammaContrastGeneralOp", device())
        self.per_channel: bool = per_channel

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 gammas: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        gamma_channel_size: int = 1
        if self.per_channel:
            gamma_channel_size = images[0].shape()[2]

        assert len(gammas) == batch_size * \
            gamma_channel_size, "The gamma number for gamma contrast should be equal to batch size if not per channel or batch size times channel size if per channel."

        return self.op.process(images, gammas, self.per_channel, sync)


class GammaContrastOp:
    """ Apply gamma contrast on input images, i.e. for each pixel value v: 255*((v/255)**gamma)
    """

    def __init__(self, device: Any, per_channel: bool = False) -> None:
        """ Initialize GammaContrastOp

        Args:
            device (Any) : the matx device used for the operation
            per_channel (bool, optional) : For each pixel, whether to apply the gamma contrast with different gamma value (True),
                                           or through out the channels using same gamma value (False). False by default.
        """
        self.op: _GammaContrastOpImpl = matx.script(_GammaContrastOpImpl)(device, per_channel)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 gammas: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply gamma contrast on input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            gammas (List[float]) : gamma value for each image / channel. If `per_channel` is False, the list should have the same size as batch size.
                                   If `per_channel` is True, the list should contain channel * batch_size elements.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.
        Returns:
            List[matx.runtime.NDArray]: converted images

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import GammaContrastOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> gammas = [0.5, 0.9, 1.2]

        >>> op = GammaContrastOp(device)
        >>> ret = op(nds, gammas)
        """
        return self.op(images, gammas, sync)
