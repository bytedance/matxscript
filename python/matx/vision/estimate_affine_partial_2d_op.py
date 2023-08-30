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

from tokenize import Double
from typing import Any, List, Tuple
from .constants._sync_mode import ASYNC

from ..native import make_native_object

import sys
matx = sys.modules['matx']


class _EstimateAffinePartial2DOpImpl:
    """ EstimateAffinePartial2DOp Impl """

    def __init__(self, device: Any) -> None:
        self.op: matx.NativeObject = make_native_object(
            "VisionEstimateAffinePartial2DGeneralOp", device())

    def __call__(self,
                 from_: matx.runtime.NDArray,
                 to: matx.runtime.NDArray,
                 method: int,
                 ransacReprojThreshold: float,
                 maxIters: int,
                 confidence: float,
                 refineIters: int) -> Tuple[matx.runtime.NDArray,
                                            matx.runtime.NDArray]:
        return self.op.process(
            from_,
            to,
            method,
            ransacReprojThreshold,
            maxIters,
            confidence,
            refineIters)


class EstimateAffinePartial2DOp:
    """ XXXX.
    """

    def __init__(self, device: Any) -> None:
        """ Initialize EstimateAffinePartial2DOp

        Args:
            device (Any) : the matx device used for the operation\
        """
        self.op_impl: _EstimateAffinePartial2DOpImpl = matx.script(
            _EstimateAffinePartial2DOpImpl)(device=device)

    def __call__(self,
                 from_: matx.runtime.NDArray,
                 to: matx.runtime.NDArray,
                 method: int = 8,
                 ransacReprojThreshold: float = 3,
                 maxIters: int = 2000,
                 confidence: float = 0.99,
                 refineIters: int = 10) -> Tuple[matx.runtime.NDArray,
                                                 matx.runtime.NDArray]:
        return self.op_impl(
            from_,
            to,
            method,
            ransacReprojThreshold,
            maxIters,
            confidence,
            refineIters)
