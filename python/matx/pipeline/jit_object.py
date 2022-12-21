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

import json
from typing import Tuple, List
from .ops import OpKernel
from ..runtime import _ffi_api as runtime_api
from .._ffi.error import trans_exception_from_c_to_py


class FuncParam:
    __slots__ = ["name", "type_code"]

    def __init__(self, name: str, type_code: int):
        self.name = name
        self.type_code = type_code

    def toobject(self):
        return {b"name": self.name.encode(), b"type_code": self.type_code}


class FuncMeta:
    __slots__ = ["name", "bound_self", "_args", "defaults"]

    def __init__(self, name: str, bound_self: bool, _args: List[FuncParam], defaults: List[object]):
        self.name = name
        self.bound_self = bound_self
        self._args = _args
        self.defaults = defaults

    def toobject(self):
        args_d = [a.toobject() for a in self._args]
        return {
            b"name": self.name.encode(),
            b"bound_self": self.bound_self,
            b"args": args_d,
            b"defaults": self.defaults}


class ClassMeta:
    __slots__ = ["name", "len_slots", "init_func", "init_args", "member_funcs"]

    def __init__(self, name: str,
                 len_slots: int,
                 init_func: FuncMeta,
                 init_args: List[object],
                 member_funcs: List[FuncMeta]):
        self.name = name
        self.len_slots = len_slots
        self.init_func = init_func
        self.init_args = init_args
        self.member_funcs = member_funcs

    def toobject(self):
        # convert init_args
        init_args = []
        for a in self.init_args:
            init_args.append(a)
        return {b"name": self.name.encode(),
                b"len_slots": self.len_slots,
                b"init_func": self.init_func.toobject(),
                b"init_args": init_args,
                b"member_funcs": [mf.toobject() for mf in self.member_funcs]}


class JitObject(OpKernel):
    """JitObject constructor.

    Parameters
    ----------
    dso_path : str
        libdso abs path
    dso_path_cxx11 : str
        libdso_cxx11 abs path
    meta_info : Optional[FuncMeta, ClassMeta]
        global function signature or class signature
    function_mapping : Dict[str, str]
        a map from raw function name to compiled function name
    """

    @trans_exception_from_c_to_py
    def __init__(self, dso_path=None,
                 dso_path_cxx11=None,
                 meta_info=None,
                 need_bundle=None,
                 function_mapping=None,
                 share=True,
                 captures=None,
                 py_source_file=b"",
                 py_source_line=-1):
        assert isinstance(function_mapping, dict)
        assert isinstance(meta_info, (ClassMeta, FuncMeta))
        if need_bundle is None:
            need_bundle = []
        else:
            need_bundle = [x.encode() for x in need_bundle]
        if captures is None:
            captures = []
        if isinstance(meta_info, ClassMeta):
            super(JitObject, self).__init__("JitObject",
                                            dso_path=dso_path.encode(),
                                            dso_path_cxx11=dso_path_cxx11.encode(),
                                            class_info=meta_info.toobject(),
                                            need_bundle=need_bundle,
                                            share=share,
                                            captures=captures,
                                            py_source_file=py_source_file,
                                            py_source_line=py_source_line)
        else:
            super(JitObject, self).__init__("JitObject",
                                            dso_path=dso_path.encode(),
                                            dso_path_cxx11=dso_path_cxx11.encode(),
                                            func_info=meta_info.toobject(),
                                            need_bundle=need_bundle,
                                            share=share,
                                            captures=captures,
                                            py_source_file=py_source_file,
                                            py_source_line=py_source_line)
        # TODO: add magic number
        self.function_mapping = function_mapping
        self.op_mapping_2_71828182846 = dict()

    def get_function(self, name):
        """Get compiled packed function. TODO: add magic number

        Parameters
        ----------
        name : str
            function name

        Returns
        -------
        func : PackedFunc
            compiled packed function

        """
        return runtime_api.JitObject_GetFunction(self.native_op, name)

    def __call__(self, *args, **kwargs):
        # TODO: fix this
        if hasattr(self, "native_call_method"):
            return self.native_call_method(*args, **kwargs)
        else:
            raise NotImplementedError("native_call_method")


def _make_user_func(raw_name, pf_func):
    pf_func.__name__ = raw_name
    return pf_func


def restore_user_behavior(ud: JitObject,
                          name: str,
                          is_class: bool,
                          init_schema: dict,
                          members: list = None):
    if is_class:
        r_map = {v: k for k, v in ud.function_mapping.items()}
        if members is not None:
            for fm in members:
                func_name = fm.name
                raw_name = r_map[func_name]
                if raw_name == "__init__":
                    continue

                if raw_name == '__call__':
                    raw_name = 'native_call_method'

                pf_func = JitOpImpl(main_func_name=func_name, jit_object=ud)
                user_func = _make_user_func(raw_name, pf_func)

                # rebound the user function
                setattr(ud, raw_name, user_func)
                # bound the JitOpImpl into JitObject
                ud.op_mapping_2_71828182846[raw_name] = pf_func
    else:
        func_name = init_schema.name
        pf_func = ud.get_function(func_name)

        def user_func(*args, **kwargs):
            assert len(kwargs) == 0
            return pf_func(*args)

        user_func.__name__ = name
        ud.native_call_method = user_func
    return ud


class JitOpImpl(OpKernel):
    """JitOpImpl constructor.

    Parameters
    ----------
    main_func_name : str
        The entry function name in JitObject
    jit_object : Optional[JitObject]
        The Compiled JitObject
    """

    def __init__(self, main_func_name=None, jit_object=None):
        assert isinstance(jit_object, JitObject)
        if main_func_name in jit_object.function_mapping:
            main_func_name = jit_object.function_mapping[main_func_name]
        super(JitOpImpl, self).__init__("JitOp",
                                        jit_object_name=jit_object.name.encode(),
                                        main_func_name=main_func_name.encode())

    @trans_exception_from_c_to_py
    def __call__(self, *args):
        """Process input values

        Parameters
        ----------
        args : Tuple[object, ...]
            op input args

        Returns
        -------
        result : object or Tuple[object, ...]
            The result data

        """
        return super(JitOpImpl, self).__call__(*args)
