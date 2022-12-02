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


class _Conv2dOpImpl:
    """ Impl: Apply conv kernels on input images.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_DEFAULT) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionConv2dGeneralOp", device())
        self.anchor: Tuple[int, int] = (-1, -1)
        self.pad_type: str = pad_type

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 kernels: List[List[List[float]]],
                 anchors: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert len(
            kernels) == batch_size, "The kernels number for conv2d should be equal to batch size."
        if len(anchors) != 0 and len(anchors) != batch_size:
            assert False, "The target size for anchors should either be empty (which will use default value (-1, -1)), or its size should be equal to batch size"

        ksize_ = matx.List()
        ksize_.reserve(batch_size)
        kernels_ = matx.List()
        kernels_.reserve(batch_size)

        for i in range(batch_size):
            cur_kernel: List = kernels[i]
            row_num: int = len(cur_kernel)
            col_num: int = len(cur_kernel[0])
            ksize_.append([col_num, row_num])
            tmp_kernel = []
            tmp_kernel.reserve(row_num * col_num)
            for row in range(row_num):
                for col in range(col_num):
                    tmp_kernel.append(cur_kernel[row][col])
            kernels_.append(tmp_kernel)

        if len(anchors) == 0:
            anchors = [self.anchor for _ in range(batch_size)]

        return self.op.process(
            images,
            kernels_,
            ksize_,
            anchors,
            self.pad_type,
            sync)


class Conv2dOp:
    """ Apply conv kernels on input images.
    """

    def __init__(self,
                 device: Any,
                 pad_type: str = BORDER_DEFAULT) -> None:
        """ Initialize Conv2dOp

        Args:
            device (Any) : the matx device used for the operation
            pad_type (str, optional) : pixel extrapolation method, if border_type is BORDER_CONSTANT, 0 would be used as border value.
        """
        self.op: _Conv2dOpImpl = matx.script(_Conv2dOpImpl)(device, pad_type)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 kernels: List[List[List[float]]],
                 anchors: List[Tuple[int, int]] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply conv kernels on input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            kernels (List[List[List[float]]]): conv kernels for each image.
            anchors (List[Tuple[int, int]], optional): anchors of each kernel, each item in this list should be a 2 element tuple (x, y).
                                                       If not given, -1 would be used by default to indicate anchor for from the center.
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
        >>> from matx.vision import Conv2dOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> # create parameters for conv2d
        >>> kernel = [[1.0/25] * 5 for _ in range(5)]
        >>> kernels = [kernel, kernel, kernel]

        >>> op = Conv2dOp(device)
        >>> ret = op(nds, kernels)
        """
        return self.op(images, kernels, anchors, sync)


class _SharpenOpImpl:
    """ Impl: Sharpen images and alpha-blend the result with the original input images.
        Sharpen kernel is [[-1, -1, -1], [-1, 8+l,-1], [-1, -1, -1]], sharpen lightness is controlled by l here.
    """

    def __init__(self,
                 device: Any,
                 alpha: float = 1.0,
                 lightness: float = 1.0,
                 pad_type: str = BORDER_DEFAULT) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionConv2dGeneralOp", device())
        self.anchor: Tuple[int, int] = (-1, -1)
        self.ksize: List[int] = [3, 3]
        self.pad_type: str = pad_type
        self.alpha: float = alpha
        self.lightness: float = lightness

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 alpha: List[float] = [],
                 lightness: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        if len(alpha) != 0 and len(alpha) != batch_size:
            assert False, "The size of alpha should be 0 or equal to batch size."
        if len(lightness) != 0 and len(lightness) != batch_size:
            assert False, "The size of lightness should be 0 or equal to batch size."

        ksize_ = matx.List()
        ksize_.reserve(batch_size)
        kernels_ = matx.List()
        kernels_.reserve(batch_size)
        anchor_ = matx.List()
        anchor_.reserve(batch_size)

        if len(alpha) == 0:
            alpha = [self.alpha for i in range(batch_size)]
        if len(lightness) == 0:
            lightness = [self.lightness for i in range(batch_size)]

        for i in range(batch_size):
            ksize_.append(self.ksize)
            anchor_.append(self.anchor)
            cur_kernel = [-alpha[i]] * 4 + [1 + 7 * alpha[i] +
                                            lightness[i] * alpha[i]] + [-alpha[i]] * 4
            kernels_.append(cur_kernel)

        return self.op.process(images, kernels_, ksize_, anchor_, self.pad_type, sync)


class SharpenOp:
    """ Sharpen images and alpha-blend the result with the original input images.
        Sharpen kernel is [[-1, -1, -1], [-1, 8+l,-1], [-1, -1, -1]], sharpen lightness is controlled by l here.
    """

    def __init__(self,
                 device: Any,
                 alpha: float = 1.0,
                 lightness: float = 1.0,
                 pad_type: str = BORDER_DEFAULT) -> None:
        """ Initialize SharpenOp

        Args:
            device (Any) : the matx device used for the operation
            alpha (float, optional) : alpha-blend factor, 1.0 by default, which means only keep the sharpened image.
            lightness (float, optional) : lightness/brightness of the sharpened image, 1.0 by default.
            pad_type (str, optional) : pixel extrapolation method, if border_type is BORDER_CONSTANT, 0 would be used as border value.
        """
        self.op: _SharpenOpImpl = matx.script(_SharpenOpImpl)(device, alpha, lightness, pad_type)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 alpha: List[float] = [],
                 lightness: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """
        Sharpen images and alpha-blend the result with the original input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            alpha (List[float], optional): blending factor for each image. If omitted, the alpha set in op initialization would be used for all images.
            lightness (List[float], optional): lightness/brightness for each image. If omitted, the lightness set in op initialization would be used for all images.
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
        >>> from matx.vision import SharpenOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> # create parameters for sharpen
        >>> alpha = [0.1, 0.5, 0.9]
        >>> lightness = [0, 1, 2]

        >>> op = SharpenOp(device)
        >>> ret = op(nds, alpha, lightness)
        """
        return self.op(images, alpha, lightness, sync)


class _EmbossOpImpl:
    """ Impl: Emboss images and alpha-blend the result with the original input images.
        Emboss kernel is [[-1-s, -s, 0], [-s, 1, s], [0, s, 1+s]], emboss strength is controlled by s here.
    """

    def __init__(self,
                 device: Any,
                 alpha: float = 1.0,
                 strength: float = 0.0,
                 pad_type: str = BORDER_DEFAULT) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionConv2dGeneralOp", device())
        self.anchor: Tuple[int, int] = (-1, -1)
        self.ksize: List[int] = [3, 3]
        self.pad_type: str = pad_type
        self.alpha: float = alpha
        self.strength: float = strength

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 alpha: List[float] = [],
                 strength: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        if len(alpha) != 0 and len(alpha) != batch_size:
            assert False, "The size of alpha should be 0 or equal to batch size."
        if len(strength) != 0 and len(strength) != batch_size:
            assert False, "The size of strength should be 0 or equal to batch size."

        ksize_ = matx.List()
        ksize_.reserve(batch_size)
        kernels_ = matx.List()
        kernels_.reserve(batch_size)
        anchor_ = matx.List()
        anchor_.reserve(batch_size)

        if len(alpha) == 0:
            alpha = [self.alpha for i in range(batch_size)]
        if len(strength) == 0:
            strength = [self.strength for i in range(batch_size)]

        for i in range(batch_size):
            ksize_.append(self.ksize)
            anchor_.append(self.anchor)
            tmp = alpha[i] * strength[i]
            cur_kernel = [-alpha[i] - tmp, -tmp, 0, -
                          tmp, 1, tmp, 0, tmp, alpha[i] + tmp]
            kernels_.append(cur_kernel)

        return self.op.process(images, kernels_, ksize_, anchor_, self.pad_type, sync)


class EmbossOp:
    """ Emboss images and alpha-blend the result with the original input images.
        Emboss kernel is [[-1-s, -s, 0], [-s, 1, s], [0, s, 1+s]], emboss strength is controlled by s here.
    """

    def __init__(self,
                 device: Any,
                 alpha: float = 1.0,
                 strength: float = 0.0,
                 pad_type: str = BORDER_DEFAULT) -> None:
        """ Initialize EmbossOp

        Args:
            device (Any) : the matx device used for the operation
            alpha (float, optional) : alpha-blend factor, 1.0 by default, which means only keep the emboss image.
            strength (float, optional) : strength of the emboss, 0.0 by default.
            pad_type (str, optional) : pixel extrapolation method, if border_type is BORDER_CONSTANT, 0 would be used as border value.
        """
        self.op: _EmbossOpImpl = matx.script(_EmbossOpImpl)(device, alpha, strength, pad_type)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 alpha: List[float] = [],
                 strength: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """
        Emboss images and alpha-blend the result with the original input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            alpha (List[float], optional): blending factor for each image. If omitted, the alpha set in op initialization would be used for all images.
            strength (List[float], optional): parameter that controls the strength of the emboss. If omitted, the strength set in op initialization would be used for all images.
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
        >>> from matx.vision import EmbossOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> # create parameters for sharpen
        >>> alpha = [0.1, 0.5, 0.9]
        >>> strength = [0, 1, 2]

        >>> op = EmbossOp(device)
        >>> ret = op(nds, alpha, strength)
        """
        return self.op(images, alpha, strength, sync)


class _EdgeDetectOpImpl:
    """ Impl: Generate a black & white edge image and alpha-blend it with the input image.
        Edge detect kernel is [[0, 1, 0], [1, -4, 1], [0, 1, 0]].
    """

    def __init__(self,
                 device: Any,
                 alpha: float = 1.0,
                 pad_type: str = BORDER_DEFAULT) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionConv2dGeneralOp", device())
        self.anchor: Tuple[int, int] = (-1, -1)
        self.ksize: List[int] = [3, 3]
        self.pad_type: str = pad_type
        self.alpha: float = alpha

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 alpha: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        if len(alpha) != 0 and len(alpha) != batch_size:
            assert False, "The size of alpha should be 0 or equal to batch size."

        ksize_ = matx.List()
        ksize_.reserve(batch_size)
        kernels_ = matx.List()
        kernels_.reserve(batch_size)
        anchor_ = matx.List()
        anchor_.reserve(batch_size)

        if len(alpha) == 0:
            alpha = [self.alpha for i in range(batch_size)]

        for i in range(batch_size):
            ksize_.append(self.ksize)
            anchor_.append(self.anchor)
            cur_kernel = [0, alpha[i], 0, alpha[i], 1 -
                          5 * alpha[i], alpha[i], 0, alpha[i], 0]
            kernels_.append(cur_kernel)

        return self.op.process(images, kernels_, ksize_, anchor_, self.pad_type, sync)


class EdgeDetectOp:
    """ Generate a black & white edge image and alpha-blend it with the input image.
        Edge detect kernel is [[0, 1, 0], [1, -4, 1], [0, 1, 0]].
    """

    def __init__(self,
                 device: Any,
                 alpha: float = 1.0,
                 pad_type: str = BORDER_DEFAULT) -> None:
        """ Initialize EdgeDetectOp

        Args:
            device (Any) : the matx device used for the operation
            alpha (float, optional) : alpha-blend factor, 1.0 by default, which means only keep the edge image.
            pad_type (str, optional) : pixel extrapolation method, if border_type is BORDER_CONSTANT, 0 would be used as border value.
        """
        self.op: _EdgeDetectOpImpl = matx.script(_EdgeDetectOpImpl)(device, alpha, pad_type)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 alpha: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """
        Generate an edge image and alpha-blend it with the input image.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            alpha (List[float]): blending factor for each image. If omitted, the alpha set in op initialization would be used for all images.
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
        >>> from matx.vision import EdgeDetectOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]

        >>> # create parameters for sharpen
        >>> alpha = [0.1, 0.5, 0.9]

        >>> op = EdgeDetectOp(device)
        >>> ret = op(nds, alpha)
        """
        return self.op(images, alpha, sync)
