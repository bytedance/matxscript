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


class _NormalizeOpImpl:
    """ NormalizeOp Impl """

    def __init__(self,
                 device: Any,
                 mean: List[float],
                 std: List[float],
                 dtype: str = "float32",
                 global_shift: float = 0.0,
                 global_scale: float = 1.0) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionNormalizeGeneralOp", mean, std, global_shift, global_scale, dtype, device())

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        return self.op.process(images, sync)


class NormalizeOp:
    """ Normalize images with mean and std, and cast the image data type to target type.
    """

    def __init__(self,
                 device: Any,
                 mean: List[float],
                 std: List[float],
                 dtype: str = "float32",
                 global_shift: float = 0.0,
                 global_scale: float = 1.0) -> None:
        """ Initialize NormalizeOp

        Args:
            device (Any) : the matx device used for the operation
            mean (List[float]) : mean for normalize
            std (List[float]) : std for normalize
            dtype (str, optional) : output data type when normalize finished, float32 by default.
            global_shift (float, optional) : shift value for all pixels after the normalization, 0.0 by default.
            global_scale (float, optional) : scale factor value for all pixels after the normalization, 1.0 by default.
        """
        self.op_impl: _NormalizeOpImpl = matx.script(_NormalizeOpImpl)(device=device,
                                                                       mean=mean,
                                                                       std=std,
                                                                       dtype=dtype,
                                                                       global_shift=global_shift,
                                                                       global_scale=global_scale)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Normalize images with mean and std, and cast the image data type to target type.

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
        >>> from matx.vision import NormalizeOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        >>> std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        >>> op = NormalizeOp(device, mean, std)
        >>> ret = op(nds)
        """
        return self.op_impl(images, sync)


class _TransposeNormalizeOpImpl:
    """ TransposeNormalizeOp Impl """

    def __init__(self,
                 device: Any,
                 mean: List[float],
                 std: List[float],
                 input_layout: str,
                 output_layout: str,
                 dtype: str = "float32",
                 global_shift: float = 0.0,
                 global_scale: float = 1.0) -> None:
        self.normalize: matx.NativeObject = make_native_object(
            "VisionNormalizeGeneralOp", mean, std, global_shift, global_scale, dtype, device())
        self.transpose: matx.NativeObject = make_native_object(
            "VisionTransposeGeneralOp", device())
        self.stack: matx.NativeObject = make_native_object(
            "VisionStackGeneralOp", device())
        self.input_layout: str = input_layout
        self.output_layout: str = output_layout

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        norm_nds = self.normalize.process(images, ASYNC)
        stack_nd = self.stack.process(norm_nds, ASYNC)
        transpose_nd = self.transpose.process(stack_nd, self.input_layout, self.output_layout, sync)
        return transpose_nd


class TransposeNormalizeOp:
    """ Normalize images with mean and std, cast the image data type to target type,
        stack the images into a single array, and then update the array format (e.g. NHWC or NCHW).
    """

    def __init__(self,
                 device: Any,
                 mean: List[float],
                 std: List[float],
                 input_layout: str,
                 output_layout: str,
                 dtype: str = "float32",
                 global_shift: float = 0.0,
                 global_scale: float = 1.0) -> None:
        """ Initialize TransposeNormalizeOp

        Args:
            device (Any) : the matx device used for the operation
            mean (List[float]) : mean for normalize
            std (List[float]) : std for normalize
            input_layout (str) : the data layout format after the stack, e.g. NHWC
            output_layout (str) : the target data layout, e.g. NCHW.
            dtype (str, optional) : output data type when normalize finished, float32 by default.
            global_shift (float, optional) : shift value for all pixels after the normalization, 0.0 by default.
            global_scale (float, optional) : scale factor value for all pixels after the normalization, 1.0 by default.
        """
        self.op_impl: _TransposeNormalizeOpImpl = matx.script(_TransposeNormalizeOpImpl)(
            device=device,
            mean=mean,
            std=std,
            input_layout=input_layout,
            output_layout=output_layout,
            dtype=dtype,
            global_scale=global_scale,
            global_shift=global_shift)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 sync: int = ASYNC) -> matx.runtime.NDArray:
        """ Normalize images with mean and std, cast the image data type to target type,
        stack the images into a single array, and then update the array format (e.g. NHWC or NCHW).

        Args:
            images (List[matx.runtime.NDArray]): target images.
            sync (int, optional): sync mode after calculating the output. when device is cpu, the params makes no difference.
                                    ASYNC -- If device is GPU, the whole calculation process is asynchronous.
                                    SYNC -- If device is GPU, the whole calculation will be blocked until this operation is finished.
                                    SYNC_CPU -- If device is GPU, the whole calculation will be blocked until this operation is finished, and the corresponding CPU array would be created and returned.
                                  Defaults to ASYNC.
        Returns:
            matx.runtime.NDArray: converted images

        Example:
        >>> import cv2
        >>> import matx
        >>> from matx.vision import TransposeNormalizeOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        >>> std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        >>> input_layout = matx.vision.NHWC
        >>> output_layout = matx.vision.NCHW

        >>> op = TransposeNormalizeOp(device, mean, std, input_layout, output_layout)
        >>> ret = op(nds)

        """
        return self.op_impl(images, sync)
