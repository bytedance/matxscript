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
import random
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _MixupImagesOpImpl:
    """ MixupImages Impl """

    def __init__(self, device: Any) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionMixupImagesGeneralOp", device())

    def __call__(self,
                 images1: List[matx.runtime.NDArray],
                 images2: List[matx.runtime.NDArray],
                 factor1: List[float],
                 factor2: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        return self.op.process(images1, images2, factor1, factor2, sync)


class MixupImagesOp:
    """ Weighted add up two images, i.e. calculate a * img1 + b * img2.
        img2 should have the same width and height as img1, while img2 would either have the same
        channel size as img1, or img2 only contains 1 channel.
    """

    def __init__(self, device: Any) -> None:
        """ Initialize MixupImagesOp

        Args:
            device (Any) : the matx device used for the operation
        """
        self.op: _MixupImagesOpImpl = matx.script(_MixupImagesOpImpl)(device)

    def __call__(self,
                 images1: List[matx.runtime.NDArray],
                 images2: List[matx.runtime.NDArray],
                 factor1: List[float],
                 factor2: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Weighted add up two images.

        Args:
            images1 (List[matx.runtime.NDArray]) : augend images.
            images2 (List[matx.runtime.NDArray]) : addend images.
            factor1 (List(float)) : weighted factor for images1.
            factor2 (List(float)) : weighted factor for images2.
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
        >>> from matx.vision import MixupImagesOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds1 = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> nds2 = [matx.array.from_numpy(image_gray, device_str) for _ in range(batch_size)]
        >>> factor1 = [0.5, 0.4, 0.3]
        >>> factor2 = [1 - f for f in factor1]

        >>> op = MixupImagesOp(device)
        >>> ret = op(nds1, nds2, factor1, factor2)
        """
        return self.op(images1, images2, factor1, factor2, sync)
