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

from typing import List, Any, Tuple
from .constants._sync_mode import ASYNC
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _ImdecodeOpImpl:
    """ Decode binary image impl"""

    def __init__(self, device: Any, fmt: str, pool_size: int = 8) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionImdecodeGeneralOp", fmt, pool_size, device())

    def __call__(self, images: List[bytes], sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        return self.op.process(images, sync)


class ImdecodeOp:
    """ Decode binary image """

    def __init__(self, device: Any, fmt: str, pool_size: int = 8) -> None:
        """ Initialize ImdecodeOp

        Args:
            device (matx.Device): device used for the operation
            fmt (str): the color type for output image, support "RGB" and "BGR"
            pool_size (int, optional): concurrency of decode operation, only for gpu, Defaults to 8.
        """
        self.op: _ImdecodeOpImpl = matx.script(
            _ImdecodeOpImpl)(device, fmt, pool_size)

    def __call__(self, images: List[bytes], sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """

        Args:
            images (List[bytes]): list of binary images
            sync (int, optional): sync mode after calculating the output. when device is cpu, the param makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will bolcking util the compute is completed.
                                    SYNC_CPU -- If device is GPU, the whole calculation will bolcking util the compute is completed, then copying the CUDA data to CPU.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: decoded images

        Examples:

        >>> import matx
        >>> from matx.vision import ImdecodeOp
        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> fd = open("./origin_image.jpeg", "rb")
        >>> content = fd.read()
        >>> fd.close()
        >>> device = matx.Device("gpu:0")
        >>> decode_op = ImdecodeOp(device, "BGR")
        >>> r = decode_op([content])
        >>> r[0].shape()
        [360, 640, 3]
        """
        return self.op(images, sync)


class _ImdecodeRandomCropOpImpl:
    """ Decode binary image and random crop impl """

    def __init__(self, device: Any, fmt: str, scale: List, ratio: List, pool_size: int = 8) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionImdecodeRandomCropGeneralOp", fmt, scale, ratio, pool_size, device())

    def __call__(self, images: List[bytes], sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        return self.op.process(images, sync)


class ImdecodeRandomCropOp:
    """ Decode binary image and random crop """

    def __init__(self, device: Any, fmt: str, scale: List, ratio: List, pool_size: int = 8) -> None:
        """

        Args:
            device (matx.Device): device used for the operation
            fmt (str): the color type for output image, support "RGB" and "BGR"
            scale (List): Specifies the lower and upper bounds for the random area of
                          the crop, before resizing. The scale is defined with respect
                          to the area of the original image.
            ratio (List): lower and upper bounds for the random aspect ratio of the crop,
                          before resizing.
            pool_size (int, optional): concurrency of decode operation, only for gpu, Defaults to 8.
        """
        self.op: _ImdecodeRandomCropOpImpl = matx.script(
            _ImdecodeRandomCropOpImpl)(device, fmt, scale, ratio, pool_size)

    def __call__(self, images: List[bytes], sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """

        Args:
            images (List[bytes]): list of binary images
            sync (int, optional): sync mode after calculating the output. when device is cpu, the param makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will bolcking util the compute is completed.
                                    SYNC_CPU -- If device is GPU, the whole calculation will bolcking util the compute is completed, then copying the CUDA data to CPU.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: decoded images

        Examples:

        >>> import matx
        >>> from matx.vision import ImdecodeRandomCropOp
        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> fd = open("./origin_image.jpeg", "rb")
        >>> content = fd.read()
        >>> fd.close()
        >>> device = matx.Device("gpu:0")
        >>> decode_op = ImdecodeRandomCropOp(device, "BGR", [0.08, 1.0], [3/4, 4/3])
        >>> ret = decode_op([content]
        >>> ret[0].shape()
        [225, 292, 3]
        """
        return self.op(images, sync)


class _ImdecodeNoExceptionOpImpl:
    """ Decode binary image without raising exception when handle invalid image impl"""

    def __init__(self, device: Any, fmt: str, pool_size: int = 8) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionImdecodeNoExceptionGeneralOp", fmt, pool_size, device())

    def __call__(self, images: List[bytes],
                 sync: int = ASYNC) -> Tuple[List[matx.runtime.NDArray], List[int]]:
        return self.op.process(images, sync)


class ImdecodeNoExceptionOp:
    """ Decode binary image without raising exception when handle invalid image"""

    def __init__(self, device: Any, fmt: str, pool_size: int = 8) -> None:
        """ Initialize ImdecodeOp

        Args:
            device (matx.Device): device used for the operation
            fmt (str): the color type for output image, support "RGB" and "BGR"
            pool_size (int, optional): concurrency of decode operation, only for gpu, Defaults to 8.
        """
        self.op: _ImdecodeNoExceptionOpImpl = matx.script(_ImdecodeNoExceptionOpImpl)(
            device, fmt, pool_size)

    def __call__(self, images: List[bytes],
                 sync: int = ASYNC) -> Tuple[List[matx.runtime.NDArray], List[int]]:
        """

        Args:
            images (List[bytes]): list of binary images
            sync (int, optional): sync mode after calculating the output. when device is cpu, the param makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will bolcking util the compute is completed.
                                    SYNC_CPU -- If device is GPU, the whole calculation will bolcking util the compute is completed, then copying the CUDA data to CPU.
                                  Defaults to ASYNC.

        Returns:
            List[matx.runtime.NDArray]: decoded images
            List[int]: 1 means operation is successful, otherwise 0

        """
        return self.op(images, sync)


class _ImdecodeNoExceptionRandomCropOpImpl:
    """ Decode binary image and random crop impl """

    def __init__(self, device: Any, fmt: str, scale: List, ratio: List, pool_size: int = 8) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionImdecodeNoExceptionRandomCropGeneralOp", fmt, scale, ratio, pool_size, device())

    def __call__(self, images: List[bytes],
                 sync: int = ASYNC) -> Tuple[List[matx.runtime.NDArray], List[int]]:
        return self.op.process(images, sync)


class ImdecodeNoExceptionRandomCropOp:

    def __init__(self, device: Any, fmt: str, scale: List, ratio: List, pool_size: int = 8) -> None:

        self.op: _ImdecodeNoExceptionRandomCropOpImpl = matx.script(
            _ImdecodeNoExceptionRandomCropOpImpl)(device, fmt, scale, ratio, pool_size)

    def __call__(self, images: List[bytes],
                 sync: int = ASYNC) -> Tuple[List[matx.runtime.NDArray], List[int]]:
        return self.op(images, sync)
