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

class _BilateralFilterOpImpl:
    """ BilateralFilter Impl """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_DEFAULT) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionBilateralFilterGeneralOp", device())
        self.pad_type: str = pad_type

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 d: List[int],
                 sigma_color: List[float],
                 sigmas_space: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert len(d) == batch_size, \
            "The d number for bilateral filter should be equal to batch size."

        assert len(sigma_color) == batch_size, \
            "The sigma color number for bilateral filter should be equal to batch size."

        assert len(sigmas_space) == batch_size, \
            "The sigma space number for bilateral filter should be equal to batch size."

        return self.op.process(images, d, sigma_color, sigmas_space, self.pad_type, sync)


class BilateralFilterOp:
    """ Apply bilateral filter on input images.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_DEFAULT) -> None:
        """ Initialize BilateralFilterOp

        Args:
            device (Any) : the matx device used for the operation
            pad_type (str, optional) : pixel extrapolation method, if border_type is BORDER_CONSTANT, 0 would be used as border value.
        """
        self.op: _BilateralFilterOpImpl = matx.script(
            _BilateralFilterOpImpl)(device, pad_type)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 d: List[int],
                 sigma_color: List[float],
                 sigmas_space: List[float],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply bilateral filter on input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            ksizes (List[Tuple[int, int]]): conv kernel size for each image, each item in this list should be a 2 element tuple (x, y).
            sigmas (List[Tuple[float, float]]): sigma for bilateral filter, each item in this list should be a 2 element tuple (x, y).
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
        >>> from matx.vision import BilateralFilterOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> ksizes = [(3, 3), (3, 5), (5, 5)]
        >>> sigmas = [(0.1, 0.1), (0.01, 0.01), (0.01, 0.1)]

        >>> op = BilateralFilterOp(device)
        >>> ret = op(nds, ksizes, sigmas)
        """
        return self.op(images, d, sigma_color, sigmas_space, sync)
