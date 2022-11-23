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


class _SplitOpImpl:
    """ Split Impl """

    def __init__(self, device: Any) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionSplitGeneralOp", device())

    def __call__(self, image: matx.runtime.NDArray,
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        return self.op.process(image, sync)


class SplitOp:
    """ split input image along channel dimension. The input is a single image.
    """

    def __init__(self, device: Any) -> None:
        """ Initialize SplitOp

        Args:
            device (Any) : the matx device used for the operation
        """
        self.op: _SplitOpImpl = matx.script(_SplitOpImpl)(device)

    def __call__(self, image: matx.runtime.NDArray,
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ split input image along channel dimension.

        Args:
            image (matx.runtime.NDArray) : target image.
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
        >>> from matx.vision import SplitOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> nd = matx.array.from_numpy(image, device_str)

        >>> op = SplitOp(device)
        >>> ret = op(nd)
        """
        return self.op(image, sync)
