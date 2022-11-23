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
from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _SaltAndPepperOpImpl:
    """ SaltAndPepper Impl """

    def __init__(self,
                 device: Any,
                 batch_size: int,
                 noise_prob: float = 0.01,
                 salt_prob: float = 0.5,
                 per_channel: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionSaltAndPepperGeneralOp", batch_size, device())
        self.batch_size: int = batch_size
        self.per_channel: bool = per_channel
        self.noise_prob: float = noise_prob
        self.salt_prob: float = salt_prob

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 noise_probs: List[float] = [],
                 salt_probs: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert batch_size <= self.batch_size, "The number of input images should be equal or less than the given batch size."
        if len(noise_probs) != batch_size and len(noise_probs) != 0:
            assert False, "The noise prob number for sp noise should be 0 or equal to batch size."
        if len(salt_probs) != batch_size and len(salt_probs) != 0:
            assert False, "The salt prob number for sp noise should be 0 or equal to batch size."

        if len(noise_probs) == 0:
            noise_probs = [self.noise_prob for i in range(batch_size)]
        if len(salt_probs) == 0:
            salt_probs = [self.salt_prob for i in range(batch_size)]

        return self.op.process(images, noise_probs, salt_probs, self.per_channel, sync)


class SaltAndPepperOp:
    """ Apply salt and pepper noise on input images.
    """

    def __init__(self,
                 device: Any,
                 batch_size: int,
                 noise_prob: float = 0.01,
                 salt_prob: float = 0.5,
                 per_channel: bool = False) -> None:
        """ Initialize SaltAndPepperOp

        Args:
            device (Any) : the matx device used for the operation
            batch_size (int) : max batch size for sp noise op. It is required for cuda randomness initialization.
                               When actually calling this op, the input batch size should be equal to or less than this value.
            noise_prob (float, optional) : the probability for each pixel to add sp noise, range from 0 to 1, 0.01 by default, can be overridden in runtime.
            salt_prob (float, optional) : for those pixels that need to apply salt_n_pepper noise, the probability that the salt noise would be, range from 0 to 1.
                                          The pepper probability would then be (1 - salt_prob). 0.5 by default, can be overridden in runtime.
            per_channel (bool, optional) : For each pixel, whether to add the noise per channel with different value (True),
                                           or through out the channels using same value (False). False by default.
        """
        self.op: _SaltAndPepperOpImpl = matx.script(_SaltAndPepperOpImpl)(
            device, batch_size, noise_prob, salt_prob, per_channel)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 noise_probs: List[float] = [],
                 salt_probs: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Apply sp noise on input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            noise_probs (List[float], optional) : probability to add sp noise for each image. If omitted, the value set during the op initialization would be used for all images.
            salt_probs (List[float], optional) : probability to add salt noise for each image. If omitted, the value set during the op initialization would be used for all images.
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
        >>> from matx.vision import SaltAndPepperOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> noise_probs = [0.1, 0.01, 0.5]
        >>> salt_probs = [0.1, 0.5, 0.9]

        >>> op = SaltAndPepperOp(device, batch_size)
        >>> ret = op(nds, noise_probs, salt_probs)
        """
        return self.op(images, noise_probs, salt_probs, sync)


class _RandomDropoutOpImpl:
    """ RandomDropout Impl """

    def __init__(self,
                 device: Any,
                 batch_size: int,
                 prob: float = 0.01,
                 per_channel: bool = False) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionSaltAndPepperGeneralOp", batch_size, device())
        self.per_channel: bool = per_channel
        self.batch_size: int = batch_size
        self.prob: float = prob

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 probs: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        batch_size: int = len(images)
        assert batch_size <= self.batch_size, "The number of input images should be equal or less than the given batch size."
        if len(probs) != batch_size and len(probs) != 0:
            assert False, "The noise prob number for sp noise should be equal to batch size."

        if len(probs) == 0:
            probs = [self.prob for i in range(batch_size)]
        salt_probs: List[float] = [0.0 for i in range(batch_size)]

        return self.op.process(images, probs, salt_probs, self.per_channel, sync)


class RandomDropoutOp:
    """ Randomly drop out some pixels (set to 0) for input images.
    """

    def __init__(self,
                 device: Any,
                 batch_size: int,
                 prob: float = 0.01,
                 per_channel: bool = False) -> None:
        """ Initialize RandomDropoutOp

        Args:
            device (Any) : the matx device used for the operation
            batch_size (int) : max batch size for sp noise op. It is required for cuda randomness initialization.
                               When actually calling this op, the input batch size should be equal to or less than this value.
            prob (float, optional) : the probability for each pixel to be dropped out, range from 0 to 1, 0.01 by default, can be overridden in runtime.
            per_channel (bool, optional) : For each pixel, whether to drop out the value differently for each channel (True),
                                           or drop out the value through out all the channels (False). False by default.
        """
        self.op: _RandomDropoutOpImpl = matx.script(_RandomDropoutOpImpl)(
            device, batch_size, prob, per_channel)

    def __call__(self,
                 images: List[matx.runtime.NDArray],
                 probs: List[float] = [],
                 sync: int = ASYNC) -> List[matx.runtime.NDArray]:
        """ Randomly drop out some pixels (set to 0) for input images.

        Args:
            images (List[matx.runtime.NDArray]): target images.
            probs (List[float], optional) : drop out probability for each image. If omitted, the value set during the op initialization would be used for all images.
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
        >>> from matx.vision import RandomDropoutOp

        >>> # Get origin_image.jpeg from https://github.com/bytedance/matxscript/tree/main/test/data/origin_image.jpeg
        >>> image = cv2.imread("./origin_image.jpeg")
        >>> device_id = 0
        >>> device_str = "gpu:{}".format(device_id)
        >>> device = matx.Device(device_str)
        >>> # Create a list of ndarrays for batch images
        >>> batch_size = 3
        >>> nds = [matx.array.from_numpy(image, device_str) for _ in range(batch_size)]
        >>> probs = [0.1, 0.01, 0.5]

        >>> op = RandomDropoutOp(device, batch_size)
        >>> ret = op(nds, probs)
        """
        return self.op(images, probs, sync)
