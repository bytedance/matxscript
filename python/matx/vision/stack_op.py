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
from .constants._sync_mode import ASYNC
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _StackOpImpl:
    """ Impl: Stack images along first dim"""

    def __init__(self, device: Any) -> None:
        """

        Args:
            device (matx.Device): device used for the operation
        """
        self.op: matx.NativeObject = make_native_object(
            "VisionStackGeneralOp", device())

    def __call__(self, images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        return self.op.process(images, sync)


class StackOp:
    """ Stack images along first dim"""

    def __init__(self, device: Any) -> None:
        """

        Args:
            device (matx.Device): device used for the operation
        """
        self.op: _StackOpImpl = matx.script(_StackOpImpl)(device)

    def __call__(self, images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        """

        Args:
            images (List[matx.runtime.NDArray]): input images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the param makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will bolcking util the compute is completed.
                                    SYNC_CPU -- If device is GPU, the whole calculation will bolcking util the compute is completed, then copying the CUDA data to CPU.
                                  Defaults to ASYNC.

        Returns:
            matx.runtime.NDArray

        Examples:

        >>> import matx
        >>> from matx.vision import ImdecodeOp, StackOp
        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> fd = open("./origin_image.jpeg", "rb")
        >>> content = fd.read()
        >>> fd.close()
        >>> device = matx.Device("gpu:0")
        >>> decode_op = ImdecodeOp(device, "BGR")
        >>> images = decode_op([content, content])
        >>> stack_op = StackOp(device)
        >>> r = stack_op(images, sync = matx.vision.SYNC)
        >>> r.shape()
        [2, 360, 640, 3]
        """
        return self.op(images, sync)
