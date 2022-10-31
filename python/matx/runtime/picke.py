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
import warnings
from . import _ffi_api
from .object_generic import to_runtime_object
from ._container import List, Dict, Set


def serialize(v):
    warnings.warn("The function matx.serialize is deprecated.", DeprecationWarning)
    rtv = to_runtime_object(v)
    return _ffi_api.Serialize(rtv)


def from_runtime_object(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        return value
    elif isinstance(value, (bytes, bytearray)):
        return value
    elif isinstance(value, str):
        return value
    elif isinstance(value, (list, dict, tuple, set)):
        return value
    elif isinstance(value, List):
        return [from_runtime_object(x) for x in value]
    elif isinstance(value, Dict):
        items = [(from_runtime_object(k), from_runtime_object(v)) for k, v in value.items()]
        return dict(items)
    elif isinstance(value, Set):
        return set([from_runtime_object(x) for x in value])
    else:
        return value


def deserialize(s: str):
    warnings.warn("The function matx.deserialize is deprecated.", DeprecationWarning)
    rtv = _ffi_api.DeSerialize(s)
    return from_runtime_object(rtv)
