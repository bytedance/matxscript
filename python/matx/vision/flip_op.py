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

from typing import List, Any
from .constants._flip_mode import *
from .constants._sync_mode import ASYNC
import random

from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _FlipOpImpl:
    """ Flip Impl """

    def __init__(self,
                 device: Any,
                 flip_code: int = HORIZONTAL_FLIP,
                 prob: float = 1.1) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionFlipGeneralOp", device())
        self.flip_code: int = flip_code
        self.prob: float = prob

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 flip_code: List[int] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        if len(flip_code) != 0 and len(flip_code) != batch_size:
            assert False, "The flip code for flip op should either be empty (which will use op's flip code for all images, default is horizontal flip), or its size sould be equal to batch size"

        flip_code_ = flip_code
        if len(flip_code) == 0:
            flip_code_ = [self.flip_code] * batch_size
        if self.prob >= 1.0:
            return self.op.process(images, flip_code_, sync)

        for i in range(batch_size):
            if random.random() >= self.prob:
                flip_code_[i] = FLIP_NOT_APPLY
        return self.op.process(images, flip_code_, sync)


class FlipOp:
    """ Flip the given images along specified directions.
    """

    def __init__(self,
                 device: Any,
                 flip_code: int = HORIZONTAL_FLIP,
                 prob: float = 1.1) -> None:
        """ Initialize FlipOp

        Args:
            device (Any): the matx device used for the operation.
            flip_code (int optional): flip type.
                                      HORIZONTAL_FLIP -- flip horizontally,
                                      VERTICAL_FLIP -- flip vertically,
                                      DIAGONAL_FLIP -- flip horizontally and vertically,
                                      FLIP_NOT_APPLY --  keep the original
                                      HORIZONTAL_FLIP by default. Could be overriden in runtime to set for each image in the batch.
            prob (float optional): probability for flipping each image, by default flipping all images with given flip code.
        """
        self.op: _FlipOpImpl = matx.script(
            _FlipOpImpl)(device, flip_code, prob)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 flip_code: List[int] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Flip images with specified directions.

        Args:
            images (List[matx.runtime.matx.runtime.NDArray]): target images.
            flip_code (List[int], optional): flip type for each image in the batch.
                                             HORIZONTAL_FLIP -- flip horizontally,
                                             VERTICAL_FLIP -- flip vertically,
                                             DIAGONAL_FLIP -- flip horizontally and vertically,
                                             FLIP_NOT_APPLY --  keep the original
                                             If omitted, the value set in the op initialization would be used for all images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.
        Returns:
            List[matx.runtime.matx.runtime.NDArray]: converted images

        Example:

        >>> import cv2
        >>> import matx
        >>> from matx.vision import FlipOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of matx.runtime.NDArrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> flip_code = matx.vision.HORIZONTAL_FLIP

        >>> op = FlipOp(device, flip_code)
        >>> ret = op(nds)
        """
        return self.op(images, flip_code, sync)
