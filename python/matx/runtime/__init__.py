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
"""MATX runtime namespace."""

# class exposures
from .packed_func import PackedFunc
from .object import Object
from .object_generic import ObjectGeneric, ObjectTypes, to_runtime_object
from .ndarray import NDArray, DataType, DataTypeCode, MATXScriptDevice
from .module import Module

from .msgpack import dumps as msgpack_dumps
from .msgpack import loads as msgpack_loads

# function exposures
from .container import Array, Map, List, Dict, Set, Tuple
from .container import Iterator
from .container import OpaqueObject
from .container import UserData
from .container import Int32Generator
from .container import Int64Generator
from .container import Float32Generator
from .container import Float64Generator
from .container import BoolGenerator
from .container import RTValueGenerator
from .trie import Trie
from .module import load_module, system_lib
from . import _ffi_api
from ._ffi_funcs import structrual_equal, structural_hash
