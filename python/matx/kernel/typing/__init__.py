#  // Copyright 2023 ByteDance Ltd. and/or its affiliates.
#  /*
#   * Licensed to the Apache Software Foundation (ASF) under one
#   * or more contributor license agreements.  See the NOTICE file
#   * distributed with this work for additional information
#   * regarding copyright ownership.  The ASF licenses this file
#   * to you under the Apache License, Version 2.0 (the
#   * "License"); you may not use this file except in compliance
#   * with the License.  You may obtain a copy of the License at
#   *
#   *   http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing,
#   * software distributed under the License is distributed on an
#   * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   * KIND, either express or implied.  See the License for the
#   * specific language governing permissions and limitations
#   * under the License.
#   */


import numpy as np

from .broadcast import *
from .kernel_type import *
from .utils import *

int8: ScalarType = ScalarType(np.int8)
int16: ScalarType = ScalarType(np.int16)
int32: ScalarType = ScalarType(np.int32)
int64: ScalarType = ScalarType(np.int64)
uint8: ScalarType = ScalarType(np.uint8)
uint16: ScalarType = ScalarType(np.uint16)
uint32: ScalarType = ScalarType(np.uint32)
uint64: ScalarType = ScalarType(np.uint64)
float16: ScalarType = ScalarType(np.float16)
float32: ScalarType = ScalarType(np.float32)
float64: ScalarType = ScalarType(np.float64)
bool_ = None  # todo implement bool

PYTYPE_TO_KERNEL_TYPE = {
    bool: ScalarType(bool),
    int: ScalarType(int),
    float: ScalarType(float),
    np.bool_: bool_,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.intc: int32,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.uint64: uint64,
    np.uintc: uint32,
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
    np.longlong: int64,
    np.ulonglong: uint64
}
