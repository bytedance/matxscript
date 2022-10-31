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
import sys
from . import _ffi_api


def make_native_function(func_name: str):
    ud_ref = _ffi_api.Func_Get(func_name.encode("utf-8"))

    def native_func_wrapper(*args):
        return _ffi_api.Func_Call(ud_ref, *args)
    return native_func_wrapper


class NativeFunction:
    __MATX_NATIVE_FUNCTION__ = True

    def __init__(self, func_name):
        self.ud_ref = _ffi_api.Func_Get(func_name.encode("utf-8"))

    def __call__(self, *args):
        return _ffi_api.Func_Call(self.ud_ref, *args)


def load_native_function(module):
    names = _ffi_api.Func_ListNames()
    for name in names:
        res = getattr(module, name, None)
        if res:
            if isinstance(res, NativeFunction):
                continue
            else:
                raise RuntimeError("{} is aleady registered in matx.native module".format(name))
        setattr(module, name, NativeFunction(name))
