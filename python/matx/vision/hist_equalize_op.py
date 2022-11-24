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


class _HistEqualizeOpImpl:
    """ HistEqualizeOp Impl """

    def __init__(self, device: Any) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionHistEqualizeGeneralOp", device())

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        return self.op.process(images, sync)


class HistEqualizeOp:
    """ Apply histgram equalization on input images. Please refer to https://en.wikipedia.org/wiki/Histogram_equalization for more information.
    """

    def __init__(self, device: Any) -> None:
        """ Initialize HistEqualizeOp

        Args:
            device (Any) : the matx device used for the operation
        """
        self.op_impl: _HistEqualizeOpImpl = matx.script(_HistEqualizeOpImpl)(device=device)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply histgram equalization on input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
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
        >>> from matx.vision import HistEqualizeOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> op = HistEqualizeOp(device)
        >>> ret = op(nds)
        """
        return self.op_impl(images, sync)
