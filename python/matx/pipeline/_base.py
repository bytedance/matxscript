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
import os
import sys
import traceback
import warnings
from . import _ffi_api
from .._ffi import libinfo


def set_op_parallelism_threads(handle, thread_num=2, share=False):
    return _ffi_api.TXSessionSetOpParallelismThreads(handle, thread_num, share)


def get_op_parallelism_threads(handle):
    return _ffi_api.TXSessionGetOpParallelismThreads(handle)


def disable_op_parallelism(handle):
    return _ffi_api.TXSessionSetOpParallelismThreads(handle, -1)


def set_pmap_threads(handle, thread_num=8, share=False):
    return _ffi_api.TXSessionSetOpComputeThreads(handle, thread_num, share)


def get_pmap_threads(handle):
    return _ffi_api.TXSessionGetOpComputeThreads(handle)


def disable_pmap_threads(handle):
    return _ffi_api.TXSessionSetOpComputeThreads(handle, -1)


def set_apply_async_threads(handle, thread_num=2, share=False):
    return _ffi_api.TXSessionSetSchedulingThreads(handle, thread_num, share)


def get_apply_async_threads(handle):
    return _ffi_api.TXSessionGetSchedulingThreads(handle)


def disable_apply_async_threads(handle):
    return _ffi_api.TXSessionSetSchedulingThreads(handle, -1)


def make_default_session():
    default_sess_handle = _ffi_api.CreateTXSessionHandle()
    # disable threadpool for fix training
    # lazy initialize threadpool when using matx.pmap matx.apply_async
    disable_op_parallelism(default_sess_handle)
    set_apply_async_threads(default_sess_handle)
    return default_sess_handle


class TXObject(object):
    default_sess_handle = make_default_session()

    def __init__(self):
        pass


class TracerWarning(Warning):
    pass


class TracerError(RuntimeError):
    pass


def current_user_frame():
    frame = None
    try:
        tx_mod = sys.modules['matx']
        mod_path = os.path.abspath(os.path.split(os.path.realpath(tx_mod.__file__))[0])
        frames = traceback.extract_stack()
        for frame in reversed(frames):
            if not frame.filename.startswith(mod_path):
                break
    except:
        pass
    return frame


def handle_trace_warning(message):
    frame = current_user_frame()
    if frame is not None:
        warnings.warn(
            f"\nFile \"{frame.filename}\", line {frame.lineno}, in {frame.name}\n  {frame.line}"
            f"\nTracerWarning: {message}",
            category=TracerWarning,
            stacklevel=0,
        )


def handle_trace_error(message):
    frame = current_user_frame()
    if frame is not None:
        raise TracerError(
            f"\nFile \"{frame.filename}\", line {frame.lineno}, in {frame.name}\n  {frame.line}"
            f"\nTracerError: {message}"
        )
    else:
        raise TracerError(message)
