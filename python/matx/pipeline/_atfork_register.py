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
import weakref
from typing import List

from .._ffi import void_p_to_runtime
from .._ffi import to_packed_func


class AtForkTask:

    def __init__(self, handle, before, after_in_child, after_in_parent):
        self.handle = handle
        self.before = before
        self.after_in_child = after_in_child
        self.after_in_parent = after_in_parent


class AtForkRegister:
    tasks: List[AtForkTask] = []

    @staticmethod
    def register_at_fork(handle, before, after_in_child, after_in_parent):
        AtForkRegister.tasks.append(AtForkTask(handle, before, after_in_child, after_in_parent))

    @staticmethod
    def unregister_at_fork(handle):
        if handle is None:
            return
        i = None
        for i, task in enumerate(AtForkRegister.tasks):
            if task.handle == handle:
                break
        if i is not None:
            AtForkRegister.tasks.pop(i)


def session_at_fork_before(weak_obj):
    sess = weak_obj()
    if sess is not None:
        sess.at_fork_before()


def session_at_fork_after_in_parent(weak_obj):
    sess = weak_obj()
    if sess is not None:
        sess.at_fork_after_in_parent()


def session_at_fork_after_in_child(weak_obj):
    sess = weak_obj()
    if sess is not None:
        sess.at_fork_after_in_child()


def register_session_at_fork(addr, sess):
    if isinstance(addr, ctypes.c_void_p):
        addr = addr.value
    weak_obj = weakref.ref(sess)

    def before():
        session_at_fork_before(weak_obj)

    def after_in_parent():
        session_at_fork_after_in_parent(weak_obj)

    def after_in_child():
        session_at_fork_after_in_child(weak_obj)

    AtForkRegister.register_at_fork(
        handle=addr,
        before=before,
        after_in_child=after_in_child,
        after_in_parent=after_in_parent,
    )


def unregister_session_at_fork(addr):
    if isinstance(addr, ctypes.c_void_p):
        addr = addr.value
    AtForkRegister.unregister_at_fork(addr)


def atfork_before():
    for task in AtForkRegister.tasks:
        if task.before:
            task.before()


def atfork_after_in_child():
    for task in AtForkRegister.tasks:
        if task.after_in_child:
            task.after_in_child()


def atfork_after_in_parent():
    for task in AtForkRegister.tasks:
        if task.after_in_parent:
            task.after_in_parent()


# only available on Unix after Python 3.7
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(
        before=atfork_before,
        after_in_child=atfork_after_in_child,
        after_in_parent=atfork_after_in_parent,
    )
else:
    from . import _ffi_api

    _ffi_api.os_register_at_fork(
        void_p_to_runtime(ctypes.c_void_p(0)),
        to_packed_func(atfork_before),
        to_packed_func(atfork_after_in_child),
        to_packed_func(atfork_after_in_parent)
    )
