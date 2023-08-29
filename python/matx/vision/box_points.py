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


class _BoxPointsOpImpl:
    """ BoxPointsOp Impl """

    def __init__(self, device: Any) -> None:
        self.op: matx.NativeObject = make_native_object("VisionBoxPointsGeneralOp", device())

    def __call__(self, rotated_rect:Tuple[Tuple[float, float], Tuple[float, float], float] ) -> matx.runtime.NDArray:
        return self.op.process(rotated_rect)


class BoxPointsOp:
    """ XXXX.
    """

    def __init__(self, device: Any) -> None:
        """ Initialize BoxPointsOp

        Args:
            device (Any) : the matx device used for the operation\
        """
        self.op_impl: _BoxPointsOpImpl = matx.script(_BoxPointsOpImpl)(device=device)

    def __call__(self, rotated_rect:Tuple[Tuple[float, float], Tuple[float, float], float] ) -> matx.runtime.NDArray:
        return self.op_impl(rotated_rect)
