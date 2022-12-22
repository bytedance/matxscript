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
from .._ffi import void_p_to_runtime
from ._atfork_register import register_session_at_fork, unregister_session_at_fork


class TXSession:

    def __init__(self, handle=None, name=None):
        if name is None:
            name = "__global__"
        if handle is None:
            self.__c_handle = _ffi_api.CreateTXSessionHandle(name)
        else:
            self.__c_handle = handle
        self.__native_free_func = _ffi_api.FreeTXSessionHandle
        self.__backend_sess_handle = void_p_to_runtime(self.__c_handle)
        register_session_at_fork(self.__c_handle, self)

    def at_fork_before(self):
        _ffi_api.TXSessionAtForkBefore(self.__c_handle)

    def at_fork_after_in_parent(self):
        _ffi_api.TXSessionAtForkAfterInParent(self.__c_handle)

    def at_fork_after_in_child(self):
        _ffi_api.TXSessionAtForkAfterInChild(self.__c_handle)

    def __del__(self):
        unregister_session_at_fork(self.__c_handle)
        self.__native_free_func(self.__backend_sess_handle)

    def __getstate__(self):
        raise TypeError("TXSession is not picklable")

    def __setstate__(self, state):
        raise TypeError("TXSession is not picklable")

    @property
    def c_handle(self):
        return self.__c_handle

    def set_device(self, device):
        return _ffi_api.TXSessionSetDevice(self.__c_handle, device)

    def set_op_parallelism_threads(self, thread_num=2, share=False):
        return _ffi_api.TXSessionSetOpParallelismThreads(self.__c_handle, thread_num, share)

    def get_op_parallelism_threads(self):
        return _ffi_api.TXSessionGetOpParallelismThreads(self.__c_handle)

    def disable_op_parallelism(self):
        return _ffi_api.TXSessionSetOpParallelismThreads(self.__c_handle, -1)

    def set_pmap_threads(self, thread_num=8, share=False):
        return _ffi_api.TXSessionSetOpComputeThreads(self.__c_handle, thread_num, share)

    def get_pmap_threads(self):
        return _ffi_api.TXSessionGetOpComputeThreads(self.__c_handle)

    def disable_pmap_threads(self):
        return _ffi_api.TXSessionSetOpComputeThreads(self.__c_handle, -1)

    def set_apply_async_threads(self, thread_num=2, share=False):
        return _ffi_api.TXSessionSetSchedulingThreads(self.__c_handle, thread_num, share)

    def get_apply_async_threads(self):
        return _ffi_api.TXSessionGetSchedulingThreads(self.__c_handle)

    def disable_apply_async_threads(self):
        return _ffi_api.TXSessionSetSchedulingThreads(self.__c_handle, -1)


def make_default_session():
    default_sess = TXSession()
    default_sess.disable_op_parallelism()
    default_sess.set_apply_async_threads()
    return default_sess


class TXObject(object):
    default_sess = make_default_session()

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
