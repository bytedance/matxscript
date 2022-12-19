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
from typing import List, Dict, Any
import matx
from matx.vision.tv_transforms import Decode, RandomHorizontalFlip, \
    RandomResizedCrop, CenterCrop, Normalize, Stack, Transpose, Compose


class MatxImagenetVisionProcessor:
    def __init__(self, device_id: int = -1, is_train: bool = True) -> None:
        self.is_train: bool = is_train
        vision_ops: List = []
        if is_train:  # image transform for training
            vision_ops = [
                matx.script(Decode)(to_rgb=True),
                matx.script(RandomResizedCrop)(size=[224, 224], scale=(0.08, 1.0), ratio=(0.75, 1.33)),
                matx.script(RandomHorizontalFlip)(),
                matx.script(Normalize)(mean=[123.675, 116.28, 103.53],
                                       std=[58.395, 57.12, 57.375]),
                matx.script(Stack)(),
                matx.script(Transpose)()
            ]
        else:  # image transform for evaluate
            vision_ops = [
                matx.script(Decode)(to_rgb=True),
                matx.script(CenterCrop)(size=[224, 224]),
                matx.script(Normalize)(mean=[123.675, 116.28, 103.53],
                                       std=[58.395, 57.12, 57.375]),
                matx.script(Stack)(),
                matx.script(Transpose)()
            ]
        self.vision_op: Any = matx.script(Compose)(device_id, vision_ops)

    def __call__(self, images: List[bytes]) -> matx.NDArray:
        return self.vision_op(images)
