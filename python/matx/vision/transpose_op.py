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

from typing import Any
from .constants._sync_mode import ASYNC
from .constants._data_format import *
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _TransposeOpImpl:
    """ Impl: Convert image tensor layout, this operators only support gpu backend.
    """

    def __init__(self,
                 device: Any,
                 input_layout: str,
                 output_layout: str) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionTransposeGeneralOp", device())
        self.input_layout: str = input_layout
        self.output_layout: str = output_layout

    def __call__(self,
                 images: matx.runtime.NDArray,
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        return self.op.process(images, self.input_layout, self.output_layout, sync)


class TransposeOp:
    """ Convert image tensor layout, this operators only support gpu backend.
    """

    def __init__(self,
                 device: Any,
                 input_layout: str,
                 output_layout: str) -> None:
        """ Initialize TransposeOp

        Args:
            device (Any): the matx device used for the operation.
            input_layout (str): the input image tensor layout. only suppport NCHW or NHWC.
            output_layout (str): the desired image tensor layout. only support NCHW or NHWC.
        """
        self.op: _TransposeOpImpl = matx.script(
            _TransposeOpImpl)(device, input_layout, output_layout)

    def __call__(self,
                 images: matx.runtime.NDArray,
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        """ Transpose image tensor.

        Args:
            images (matx.runtime.NDArray): input images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.

        Returns:
            matx.runtime.NDArray: Transpose images.

        Example:

        >>> import cv2
        >>> import matx
        >>> import numpy as np
        >>> from matx.vision import TransposeOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> batch_image = np.stack([image, image, image, image])
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a NHWC image tensor
        >>> nds = matx.array.from_numpy(batch_image, device_str)

        >>> op = TransposeOp(device=device,
                             input_layout=matx.vision.NHWC,
                             output_layout=matx.vision.NCHW)
        >>> ret = op(nds)
        """
        return self.op(images, sync)
