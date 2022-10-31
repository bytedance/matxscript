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
from typing import List
import matx


class Pad:
    def __init__(self) -> None:
        pass

    def __call__(self,
                 images: List[matx.NDArray],
                 top_pads: List[int]) -> List[matx.NDArray]:
        return images


class PadTest:
    def __init__(self, padding: List[int]) -> None:
        self.padding: List[int] = padding
        self.op: Pad = Pad()

    def __call__(self, images: List[matx.NDArray]) -> List[matx.NDArray]:
        return self.op(images, [self.padding[0]])


if __name__ == '__main__':
    matx.script(PadTest)
