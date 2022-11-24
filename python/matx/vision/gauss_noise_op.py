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
from .opencv._cv_border_types import BORDER_DEFAULT

from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _GaussNoiseOpImpl:
    """ GaussNoise Impl """

    def __init__(self,
                 device: Any,
                 batch_size: int,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 per_channel: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionGaussNoiseGeneralOp", batch_size, device())
        self.per_channel: bool = per_channel
        self.batch_size: int = batch_size
        self.mu: float = mu
        self.sigma: float = sigma

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 mus: List[float] = [],
                 sigmas: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert batch_size <= self.batch_size, "The number of input images should be equal or less than the given batch size."
        if len(mus) != 0 and len(mus) != batch_size:
            assert False, "The mu number for gauss noise should be 0 or equal to batch size."
        if len(sigmas) != 0 and len(sigmas) != batch_size:
            assert False, "The sigma number for gauss noise should be 0 or equal to batch size."

        if len(mus) == 0:
            mus = [self.mu for i in range(batch_size)]
        if len(sigmas) == 0:
            sigmas = [self.sigma for i in range(batch_size)]

        return self.op.process(images, mus, sigmas, self.per_channel, sync)


class GaussNoiseOp:
    """ Apply gaussian noise on input images.
    """

    def __init__(self,
                 device: Any,
                 batch_size: int,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 per_channel: bool = False) -> None:
        """ Initialize GaussNoiseOp

        Args:
            device (Any) : the matx device used for the operation
            batch_size (int) : max batch size for gaussian noise op. It is required for cuda randomness initialization.
                               When actually calling this op, the input batch size should be equal to or less than this value.
            mu (float, optional) : mu for gaussian noise. It is a global value for all images, can be overridden in calling time, 0.0 by default.
            sigma (float, optional) : sigma for gaussian noise. It is a global value for all images, can be overridden in calling time, 1.0 by default.
            per_channel (bool, optional) : For each pixel, whether to add the noise per channel with different value (True),
                                           or through out the channels using same value (False). False by default.
        """
        self.op: _GaussNoiseOpImpl = matx.script(_GaussNoiseOpImpl)(
            device, batch_size, mu, sigma, per_channel)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 mus: List[float] = [],
                 sigmas: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply gaussian noise on input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            mus (List[float], optional) : mu value for each image. If omitted, the mu value set during the op initialization would be used for all images.
            sigmas (List[float], optional): sigma value for each image. If omitted, the sigma value set during the op initialization would be used for all images.
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
        >>> from matx.vision import GaussNoiseOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> mus = [0.0, 5.0, 10.0]
        >>> sigmas = [0.01, 0.1, 1]

        >>> op = GaussNoiseOp(device, batch_size)
        >>> ret = op(nds, mus, sigmas)
        """
        return self.op(images, mus, sigmas, sync)
