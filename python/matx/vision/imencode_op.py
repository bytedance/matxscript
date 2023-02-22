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


class _ImencodeOpImpl:
    """ Encode image to jpg binary impl"""

    def __init__(
            self,
            device: Any,
            fmt: str,
            quality: int,
            optimized_Huffman: bool,
            pool_size: int = 8) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionImencodeGeneralOp", fmt, quality, optimized_Huffman, pool_size, device())

    def __call__(self, images: List[matx.runtime.NDArray]) -> List[bytes]:
        return self.op.process(images)


class ImencodeOp:
    """ Encode image to jpg binary """

    def __init__(
            self,
            device: Any,
            fmt: str,
            quality: int,
            optimized_Huffman: bool,
            pool_size: int = 8) -> None:
        """ Initialize ImencodeOp

        Args:
            device (matx.Device): device used for the operation
            fmt (str): the color type for output image, support "RGB" and "BGR"
            quality (int): the jpeg quality, valid between [1, 100]. 100 means no loss.
            optimized_Huffman (bool): boolean value that control if optimized huffman tree is used.
                                      Enabling it usually means slower encoding but smaller binary size.
            pool_size (int, optional): concurrency of encode operation, only for gpu, Defaults to 8.
        """
        self.op: _ImencodeOpImpl = matx.script(
            _ImencodeOpImpl)(device, fmt, quality, optimized_Huffman, pool_size)

    def __call__(self, images: List[matx.runtime.NDArray]) -> List[bytes]:
        """ there is no sync model as all data will be on cpu before the return

        Args:
            images (List[matx.runtime.NDArray]): list of image on GPU

        Returns:
            List[bytes]: jpg encoded images

        Examples:

        >>> import matx
        >>> from matx.vision import ImencodeOp
        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_str = "gpu:0"
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)
        >>> device = matx.Device(device_str)
        >>> encode_op = ImencodeOp(device, "BGR")
        >>> r = encode_op([nds])
        """
        return self.op(images)


class _ImencodeNoExceptionOpImpl:
    """ Encode image to jpg binary without raising exception when handle invalid image impl"""

    def __init__(
            self,
            device: Any,
            fmt: str,
            quality: int,
            optimized_Huffman: bool,
            pool_size: int = 8) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionImencodeNoExceptionGeneralOp",
            fmt,
            quality,
            optimized_Huffman,
            pool_size,
            device())

    def __call__(self, images: List[matx.runtime.NDArray]
                 ) -> Tuple[List[bytes], List[int]]:
        return self.op.process(images)


class ImencodeNoExceptionOp:
    """  Encode image to jpg binary without raising exception when handle invalid image"""

    def __init__(
            self,
            device: Any,
            fmt: str,
            quality: int,
            optimized_Huffman: bool,
            pool_size: int = 8) -> None:
        """ Initialize ImencodeOp

        Args:
            device (matx.Device): device used for the operation
            fmt (str): the color type for output image, support "RGB" and "BGR"
            quality (int): the jpeg quality, valid between [1, 100]. 100 means no loss.
            optimized_Huffman (bool): boolean value that control if optimized huffman tree is used.
                                      Enabling it usually means slower encoding but smaller binary size.
            pool_size (int, optional): concurrency of encode operation, only for gpu, Defaults to 8.
        """
        self.op: _ImencodeNoExceptionOpImpl = matx.script(_ImencodeNoExceptionOpImpl)(
            device, fmt, quality, optimized_Huffman, pool_size)

    def __call__(self, images: List[matx.runtime.NDArray]
                 ) -> Tuple[List[bytes], List[int]]:
        """

        Args:
            images (List[matx.runtime.NDArray]): list of image on GPU

        Returns:
            List[bytes]: jpg encoded images

        Examples:

        >>> import matx
        >>> from matx.vision import ImencodeOp
        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_str = "gpu:0"
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)
        >>> device = matx.Device(device_str)
        >>> encode_op = ImencodeOp(device, "BGR")
        >>> r = encode_op([nds])
        """
        return self.op(images)
