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
from .opencv._cv_border_types import BORDER_CONSTANT
from .opencv._cv_interpolation_flags import INTER_LINEAR
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _WarpAffineOpImpl:
    """ Impl: Apply warp affine on images.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_CONSTANT,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 interp: str = INTER_LINEAR) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionWarpAffineGeneralOp", device())
        self.pad_type: str = pad_type
        self.pad_values: Tuple[int, int, int] = pad_values
        self.interp: str = interp

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 affine_matrix: List[List[List[float]]],
                 dsize: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert len(
            affine_matrix) == batch_size, "The matrix number for warp affine should be equal to batch size."
        if len(dsize) != 0 and len(dsize) != batch_size:
            assert False, "The target size for warp affine should either be empty (which will use the original image size as the output size), or its size should be equal to batch size"

        dsize_ = matx.List()
        dsize_.reserve(batch_size)

        if len(dsize) == batch_size:
            for i in range(batch_size):
                height: int = dsize[i][0]
                width: int = dsize[i][1]
                dsize_.append([height, width])
        else:
            for i in range(batch_size):
                image: matx.runtime.NDArray = images[i]
                height: int = image.shape()[0]
                width: int = image.shape()[1]
                dsize_.append([height, width])

        return self.op.process(
            images,
            dsize_,
            affine_matrix,
            self.pad_type,
            self.pad_values,
            self.interp,
            sync)


class WarpAffineOp:
    """ Apply warp affine on images.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_CONSTANT,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 interp: str = INTER_LINEAR) -> None:
        """ Initialize WarpAffineOp

        Args:
            device (Any) : the matx device used for the operation
            pad_type (str, optional) : border type to fill the target image, use constant value by default.
            pad_values (Tuple[int, int, int], optional) : the border value to fill the target image if pad_type is BORDER_CONSTANT, (0, 0, 0) by default.
            interp (str, optional) : desired interpolation method.
                                     INTER_NEAREST -- a nearest-neighbor interpolation;
                                     INTER_LINEAR -- a bilinear interpolation (used by default);
                                     INTER_CUBIC -- a bicubic interpolation over 4x4 pixel neighborhood;
                                     INTER_LINEAR by default.
        """
        self.op: _WarpAffineOpImpl = matx.script(
            _WarpAffineOpImpl)(device, pad_type, pad_values, interp)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 affine_matrix: List[List[List[float]]],
                 dsize: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply warp affine on images.

        Args:
            images (List[matx.runtime.NDArray]) : target images.
            affine_matrix (List[List[List[float]]]) : affine matrix for each image, each matrix should be of shape 2x3.
            dsize (List[Tuple[int, int]], optional) : target output size (h, w) for affine transformation.
                                            If omitted, the image original shape would be used.
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
        >>> from matx.vision import WarpAffineOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> affine_matrix1 = [[0, 1, 0], [-1, 0, 0]] # rotate
        >>> affine_matrix2 = [[1, 0, 10], [0, 1, 10]] # shift
        >>> affine_matrix3 = [[1, 0, 0], [0.15, 1, 0]] # shear
        >>> affine_matrix = [affine_matrix1, affine_matrix2, affine_matrix3]

        >>> op = WarpAffineOp(device)
        >>> ret = op(nds, affine_matrix)
        """
        return self.op(images, affine_matrix, dsize, sync)
