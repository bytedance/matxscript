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
import ctypes
import sys
from ._native_object import NativeObject
from ._native_object import NativeClass
from ._native_object import make_native_object_creator
from ._native_object import make_native_object
from ._native_object import load_native_object
from ._native_func import NativeFunction
from ._native_func import make_native_function
from ._native_func import load_native_function
from . import _ffi_api

_cur_module = sys.modules[__name__]


def call_native_function(func_name: str, *args):
    return _ffi_api.call_native_function(func_name, *args)


def load_native(dso_path: str = ""):
    if dso_path:
        ctypes.CDLL(dso_path)
    load_native_object(_cur_module)
    load_native_function(_cur_module)


load_native_function(_cur_module)
load_native_object(_cur_module)
