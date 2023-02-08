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

from .._ffi._selector import _set_fast_pipeline_object_converter
from .._ffi._selector import _set_class_symbol
from .symbol import BaseSymbol
from ..native import NativeObject
from .ops import OpKernel
from .jit_object import JitObject


def _pipeline_object_converter(value):
    if isinstance(value, JitObject):
        return value.native_op
    if isinstance(value, OpKernel):
        return value.native_op
    if isinstance(value, NativeObject):
        return value.ud_ref
    return value


_PipelineClasses = (JitObject, OpKernel, NativeObject,)
_set_fast_pipeline_object_converter(_PipelineClasses, _pipeline_object_converter)
_set_class_symbol(BaseSymbol)
