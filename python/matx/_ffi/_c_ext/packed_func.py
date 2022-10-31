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
# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches, global-statement, unused-import
"""Function configuration API."""
from ._loader import matx_script_api
from ..runtime_ctypes import ArgTypeCode
from ..runtime_ctypes import ObjectRValueRef
from . import types

_CLASS_MODULE = None
_CLASS_PACKED_FUNC = None
_CLASS_OBJECT_GENERIC = None
_FUNC_CONVERT_TO_OBJECT = None

PackedFuncBase = matx_script_api.PackedFuncBase


def _set_class_module(module_class):
    """Initialize the module."""
    global _CLASS_MODULE
    _CLASS_MODULE = module_class

    def _conv(mod: _CLASS_MODULE):
        handle = mod.handle
        assert isinstance(handle, int)
        return matx_script_api.make_any(ArgTypeCode.MODULE_HANDLE, 0, handle, 0)

    _register_input_callback(_CLASS_MODULE, _conv)


def _set_class_packed_func(packed_func_class):
    global _CLASS_PACKED_FUNC
    _CLASS_PACKED_FUNC = packed_func_class


def _set_class_object_generic(object_generic_class, func_convert_to_object):
    global _CLASS_OBJECT_GENERIC
    global _FUNC_CONVERT_TO_OBJECT
    _CLASS_OBJECT_GENERIC = object_generic_class
    _FUNC_CONVERT_TO_OBJECT = func_convert_to_object
    matx_script_api.register_input_callback(_CLASS_OBJECT_GENERIC, _FUNC_CONVERT_TO_OBJECT)


def _return_module(handle):
    """Return function"""
    return _CLASS_MODULE(handle)


def _handle_return_func(handle):
    """Return function"""
    return _CLASS_PACKED_FUNC(handle, False)


def _get_global_func(name, allow_missing=False):
    return matx_script_api.get_global_func(name, allow_missing)


def to_packed_func(py_func):
    """Convert a python function to Packed function

    Parameters
    ----------
    py_func : Callable
        The python function to be converted.

    Returns
    -------
    cc_func: PackedFunc
        The converted cc function.
    """
    return matx_script_api.convert_to_packed_func(py_func)


def _register_input_callback(cls, callback):
    matx_script_api.register_input_callback(cls, callback)


def _rvalue_ref_conv(rvalue: ObjectRValueRef):
    handle, code = matx_script_api.steal_object_handle(rvalue.obj)
    return matx_script_api.make_any(code, 0, handle, 1)


_register_input_callback(ObjectRValueRef, _rvalue_ref_conv)
matx_script_api.set_packedfunc_creator(_handle_return_func)
matx_script_api.register_object(ArgTypeCode.MODULE_HANDLE, _return_module)
