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

import abc
import sys
matx = sys.modules['matx']
from .. import ASYNC
import torch
import random


class BaseInterfaceClass:
    def __init__(self,
                 device_id: int = -2,
                 sync: int = ASYNC) -> None:
        self._device_id: int = device_id
        self._has_sync: bool = True
        self._sync: int = sync

    def device_id(self) -> int:
        return self._device_id

    def has_sync(self) -> bool:
        return self._has_sync

    def sync(self) -> int:
        return self._sync


class BatchBaseClass:
    def __init__(self):
        pass

    def _process(self, images: List[matx.NDArray]) -> List[matx.NDArray]:
        return images

    def _get_images_from_apply_index(
            self,
            images: List[matx.NDArray],
            apply_index: List[int]) -> List[matx.NDArray]:
        applied_size: int = len(apply_index)
        applied_images = [images[apply_index[i]] for i in range(applied_size)]
        return applied_images

    def _put_back_converted_images(
            self,
            images: List[matx.NDArray],
            apply_index: List[int],
            applied_images: List[matx.NDArray]) -> List[matx.NDArray]:
        applied_size: int = len(apply_index)
        assert applied_size == len(
            applied_images), "The size of applied images should be equal to the size of applied index."
        for i in range(applied_size):
            images[apply_index[i]] = applied_images[i]
        return images

    def __call__(
            self,
            images: List[matx.NDArray],
            apply_index: List[int]) -> List[matx.NDArray]:
        batch_size: int = len(images)
        if len(apply_index) == 0:
            for i in range(batch_size):
                apply_index.append(i)
        target_images: List[matx.NDArray] = self._get_images_from_apply_index(images, apply_index)
        new_images: List[matx.NDArray] = self._process(target_images)
        res: List[matx.NDArray] = self._put_back_converted_images(images, apply_index, new_images)
        return res


class BatchRandomBaseClass(BatchBaseClass):
    def __init__(self, prob: float) -> None:
        self.prob: float = prob

    def _random_index(self, batch_size: int) -> List[int]:
        if self.prob is None:
            return []
        apply_index = []
        for i in range(batch_size):
            if random.random() < self.prob:
                apply_index.append(i)
        return apply_index

    def __call__(
            self,
            images: List[matx.NDArray],
            apply_index: List[int]) -> List[matx.NDArray]:
        batch_size: int = len(images)
        if len(apply_index) == 0:
            apply_index = self._random_index(batch_size)
        if len(apply_index) == 0:
            return images
        target_images: List[matx.NDArray] = self._get_images_from_apply_index(images, apply_index)
        new_images: List[matx.NDArray] = self._process(target_images)
        res: List[matx.NDArray] = self._put_back_converted_images(images, apply_index, new_images)
        return res
