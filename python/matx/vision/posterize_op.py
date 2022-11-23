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


class _PosterizeOpImpl:
    """ PosterizeOp Impl """

    def __init__(self,
                 device: Any,
                 bit: int = 4,
                 prob: float = 1.1) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionPosterizeGeneralOp", device())
        self.prob: float = prob
        self.bit: int = bit

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 bits: List[int] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        if len(bits) != 0 and len(bits) != batch_size:
            assert False, "The length of bits should be equal to input images"
        if len(bits) == 0:
            bits = [self.bit for _ in range(batch_size)]
        if self.prob >= 1.0:
            return self.op.process(images, bits, sync)
        for i in range(batch_size):
            if random.random() >= self.prob:
                bits[i] = 8
        return self.op.process(images, bits, sync)


class PosterizeOp:
    """ Apply posterization on images. i.e. remove certain bits for each pixel value,
        e.g. with bit=4, pixel 77 would become 64 (the last 4 bits are set to 0).
    """

    def __init__(self,
                 device: Any,
                 bit: int = 4,
                 prob: float = 1.1) -> None:
        """ Initialize PosterizeOp

        Args:
            device (Any) : the matx device used for the operation
            bit (int, optional): bit for posterization for all images, range from [0, 8], set to 4 by default.
            prob (float, optional): probability for posterization on each image. Apply on all by default.
        """
        self.op_impl: _PosterizeOpImpl = matx.script(_PosterizeOpImpl)(device=device,
                                                                       bit=bit,
                                                                       prob=prob)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 bits: List[int] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply posterization on images. Only support uint8 images

        Args:
            images (List[matx.runtime.NDArray]): target images.
            bits (List[int]): posterization bit for each image. If not given, the bit for op initialization would be used.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import PosterizeOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> bits = [1, 4, 7]

        >>> op = PosterizeOp(device)
        >>> ret = op(nds, bits)
        """
        return self.op_impl(images, bits, sync)
