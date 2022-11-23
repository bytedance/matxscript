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


class _SolarizeOpImpl:
    """ SolarizeOp Impl """

    def __init__(self,
                 device: Any,
                 threshold: float = 128.0,
                 prob: float = 1.1) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionSolarizeGeneralOp", device())
        self.prob: float = prob
        self.threshold: float = threshold

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 threshold: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        if len(threshold) != 0 and len(threshold) != batch_size:
            assert False, "The length of threshold should be 0 or equal to input images"
        if len(threshold) == 0:
            threshold = [self.threshold for _ in range(batch_size)]
        if self.prob >= 1.0:
            return self.op.process(images, threshold, sync)
        for i in range(batch_size):
            if random.random() >= self.prob:
                threshold[i] = 256.0
        return self.op.process(images, threshold, sync)


class SolarizeOp:
    """ Apply solarization on images. i.e. invert the pixel value if the value is above the given threshold.
    """

    def __init__(self,
                 device: Any,
                 threshold: float = 128.0,
                 prob: float = 1.1) -> None:
        """ Initialize SolarizeOp

        Args:
            device (Any) : the matx device used for the operation
            threshold (float, optional): solarization threshold for all images, 128 by default.
            prob (float, optional): probability for solarization on each image. Apply on all by default.
        """
        self.op_impl: _SolarizeOpImpl = matx.script(_SolarizeOpImpl)(device=device,
                                                                     threshold=threshold,
                                                                     prob=prob)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 threshold: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply solarization on images. Only support uint8 images

        Args:
            images (List[matx.runtime.NDArray]): target images.
            threshold (List[float], optional): solarization threshold for each image. If not given the threshold for op initialization would be used.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import SolarizeOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> threshold = [80, 160, 240]

        >>> op = SolarizeOp(device)
        >>> ret = op(nds, threshold)
        """
        return self.op_impl(images, threshold, sync)
