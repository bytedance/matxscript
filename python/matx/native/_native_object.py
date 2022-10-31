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
import inspect

from . import _ffi_api
import matx.native
from typing import List, Union
import ctypes


class NativeObject(object):
    __slots__ = ["cls_name", "ud_ref"]

    def __init__(self, cls_name, *args):
        self.cls_name = cls_name
        self.ud_ref = _ffi_api.CreateNativeObject(self.cls_name.encode(), *args)


def restore_native_method(class_name: str, ud_instance: NativeObject):
    def make_attr_method(ud_instance: NativeObject, func_name: str):
        fname = func_name.encode()

        def native_method(*args):
            return _ffi_api.NativeObject_Call(ud_instance.ud_ref, fname, *args)

        native_method.__name__ = func_name
        return native_method

    func_names = _ffi_api.GetFunctionTable(class_name.encode())
    for func_name in func_names:
        func = make_attr_method(ud_instance, func_name)
        setattr(ud_instance, func_name, func)
    return ud_instance


def make_native_object_creator(class_name):
    if isinstance(class_name, (bytes, bytearray)):
        class_name = class_name.decode()
    assert isinstance(class_name, str)
    found = _ffi_api.Exist(class_name.encode())
    if not found:
        raise RuntimeError("native class not found: %s" % (class_name))

    def creator(*args):
        """Create a new resource by kwargs

        Parameters
        ----------
        *args :
            arguments of this native data.

        Returns
        -------
        result : NativeObject

        """

        class NativeObjectWrapper(NativeObject):

            def __init__(self):
                super(NativeObjectWrapper, self).__init__(class_name, *args)

        NativeObjectWrapper.__name__ = class_name
        ud = NativeObjectWrapper()
        return restore_native_method(class_name, ud)

    creator.__name__ = class_name
    return creator


def make_native_object(class_name, *args):
    is_native_op = _ffi_api.ClassNameIsNativeOp(class_name.encode())
    # assert is_native_op, "%s is NativeOp" % class_name
    creator = make_native_object_creator(class_name)
    return creator(*args)


def set_class_method(cls):
    def make_class_method(func_name):
        fn_b_name = func_name.encode()

        def class_method(self, *args):
            return _ffi_api.NativeObject_Call(self.ud_ref, fn_b_name, *args)
        class_method.__name__ = func_name
        return class_method
    func_names = _ffi_api.GetFunctionTable(cls.__name__.encode())
    for func_name in func_names:
        func = make_class_method(func_name)
        setattr(cls, func_name, func)


class NativeClass:
    __MATX_NATIVE_OBJECT__ = True

    def __init__(self, *args):
        self.ud_ref = _ffi_api.CreateNativeObject(self.__class__.__name__.encode(), *args)


def load_native_object(module):
    names = _ffi_api.ListPureObjNames()
    for name in names:
        res = getattr(module, name, None)
        if res:
            if inspect.isclass(res) and issubclass(res, NativeClass):
                continue
            else:
                raise RuntimeError("{} is aleady registered in matx.native module".format(name))
        cls = type(name, (NativeClass, ), {})
        set_class_method(cls)
        setattr(module, name, cls)
