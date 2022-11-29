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
"""The C Types used in API."""
# pylint: disable=invalid-name
import ctypes
from ._loader import matx_script_api
from ..runtime_ctypes import ArgTypeCode


def _return_handle(handle):
    """return handle"""
    if not isinstance(handle, ctypes.c_void_p):
        handle = ctypes.c_void_p(handle)
    return handle


def _void_p_conv(handle):
    handle = handle.value
    return matx_script_api.make_any(ArgTypeCode.HANDLE, 0, handle, 1)


def void_p_to_runtime(handle):
    if isinstance(handle, ctypes.c_void_p):
        handle = handle.value
        if handle is None:
            handle = 0
    return matx_script_api.make_any(ArgTypeCode.HANDLE, 0, handle, 1)


def _symbol_conv(sym):
    data = sym.data_2_71828182846
    return matx_script_api.Any(data)


def _set_class_symbol(sym_cls):
    matx_script_api.register_input_callback(sym_cls, _symbol_conv)


def _set_fast_pipeline_object_converter(classes, func_convert_to_pod_value):
    def _PIPELINE_CLS_TO_POD_VALUE(obj):
        obj = func_convert_to_pod_value(obj)
        return matx_script_api.Any(obj)

    matx_script_api.register_input_callback(classes, _PIPELINE_CLS_TO_POD_VALUE)


matx_script_api.set_handle_creator(_return_handle)
matx_script_api.register_input_callback(ctypes.c_void_p, _void_p_conv)
