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


class _WarpPerspectiveOpImpl:
    """ WarpPerspective Impl """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_CONSTANT,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 interp: str = INTER_LINEAR) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionWarpPerspectiveGeneralOp", device())
        self.pad_type: str = pad_type
        self.pad_values: Tuple[int, int, int] = pad_values
        self.interp: str = interp
        self.dsize: Tuple[int, int] = (-1, -1)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 pts: List[List[List[Tuple[float, float]]]],
                 dsize: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert len(pts) == batch_size, "The length of pts should be eauql to batch size."
        if len(dsize) != 0 and len(dsize) != batch_size:
            assert False, "The target size for warp perspective should either be empty (which will use the original image size as the output size), or its size should be equal to batch size"

        dsize_ = [self.dsize for _ in range(batch_size)]
        if len(dsize) == batch_size:
            for i in range(batch_size):
                dsize_[i] = dsize[i]

        return self.op.process(
            images,
            dsize_,
            pts,
            self.pad_type,
            self.pad_values,
            self.interp,
            sync)


class WarpPerspectiveOp:
    """ Apply warp perspective on images.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_CONSTANT,
                 pad_values: Tuple[int, int, int] = (0, 0, 0),
                 interp: str = INTER_LINEAR) -> None:
        """ Initialize WarpPerspectiveOp

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
        self.op: _WarpPerspectiveOpImpl = matx.script(
            _WarpPerspectiveOpImpl)(device, pad_type, pad_values, interp)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 pts: List[List[List[Tuple[float, float]]]],
                 dsize: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply warp perspective on images.

        Args:
            images (List[matx.runtime.NDArray]) : target images.
            pts (List[List[List[Tuple[float, float]]]]) : coordinate pairs of src and dst points.
                                                          the shape of pts is Nx2xMx2, where N is the batch size, the left side 2 represents
                                                          src and dst points respectively, M means the number of points for src/dst, the right
                                                          side 2 represents the coordinator for each point, which is a 2 element tuple (x, y).
                                                          If still confused, please see the usage in the example below.
            dsize (List[Tuple[int, int]], optional) : target output size (h, w) for perspective transformation.
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
        >>> from matx.vision import WarpPerspectiveOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> height, width = image.shape[:2]
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> src1_ptrs = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        >>> dst1_ptrs = [(0, height * 0.13), (width * 0.9, 0),
                         (width * 0.2, height * 0.7), (width * 0.8, height - 1)]
        >>> src2_ptrs = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        >>> dst2_ptrs = [(0, height * 0.03), (width * 0.93, 0),
                         (width * 0.23, height * 0.73), (width * 0.83, height - 1)]

        >>> src3_ptrs = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        >>> dst3_ptrs = [(0, height * 0.33), (width * 0.73, 0),
                         (width * 0.23, height * 0.83), (width * 0.63, height - 1)]
        >>> pts = [[src1_ptrs, dst1_ptrs], [src2_ptrs, dst2_ptrs], [src3_ptrs, dst3_ptrs]]

        >>> op = WarpPerspectiveOp(device)
        >>> ret = op(nds, pts)
        """
        return self.op(images, pts, dsize, sync)
