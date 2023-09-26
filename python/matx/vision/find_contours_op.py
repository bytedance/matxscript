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


class _FindContoursOpImpl:
    """ FindContoursOp Impl """

    def __init__(self, device: Any) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionFindContoursGeneralOp", device())

    def __call__(self, image: matx.runtime.NDArray, mode: int, method: int, offset_x: int,
                 offset_y: int) -> Tuple[Tuple[matx.runtime.NDArray], matx.runtime.NDArray]:
        return self.op.process(image, mode, method, offset_x, offset_y)


class FindContoursOp:
    """ XXXX.
    """

    def __init__(self, device: Any) -> None:
        """ Initialize FindContoursOp

        Args:
            device (Any) : the matx device used for the operation\
        """
        self.op_impl: _FindContoursOpImpl = matx.script(_FindContoursOpImpl)(device=device)

    def __call__(self, image: matx.runtime.NDArray, mode: int, method: int, offset: Tuple[int, int] = (
            0, 0)) -> Tuple[List[matx.runtime.NDArray], matx.runtime.NDArray]:
        return self.op_impl(image, mode, method, offset[0], offset[1])
