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
import os
import sys
import json
import ctypes
import traceback
import warnings
from .._ffi.base import string_types
from .._ffi import op_kernel_call
from .._ffi import void_p_to_runtime
from . import _ffi_api
from ._base import TXObject
from ._base import handle_trace_warning
from ..runtime import to_runtime_object
from .symbol import Symbol
from .symbol import Variable
from .symbol import Constant
from .symbol import BaseSymbol
from .symbol import make_symbol
from ._tracing_state import tracing
from ..runtime import Tuple
from .._ffi.runtime_ctypes import TypeIndex


def _format_trace_warning_message(arg_idx):
    arg_idx += 1
    if arg_idx == 1:
        arg_idx_s = '1st'
    elif arg_idx == 2:
        arg_idx_s = '2nd'
    elif arg_idx == 3:
        arg_idx_s = '3rd'
    else:
        arg_idx_s = f'{arg_idx}th'
    return f"The {arg_idx_s} argument will be treated as a constant in the future. " \
           f"This means that the trace might not generalize to other inputs!"


class OpKernel(TXObject):
    __MATX_NATIVE_OP__ = True

    def __init__(self, class_name, **kwargs):
        super(OpKernel, self).__init__()
        self.__holder = sys.modules['matx']
        self.__native_free_func = _ffi_api.FreeNativeOp
        self._native_class_name = class_name
        self.sess_handle = TXObject.default_sess.c_handle
        self.__backend_sess_handle = void_p_to_runtime(self.sess_handle)
        attrs = dict()
        for k, v in kwargs.items():
            attrs[k.encode()] = v
        try:
            _ffi_api.SetOpInitState(True)
            attrs = to_runtime_object(attrs)
            native_op = _ffi_api.CreateNativeOp(self.sess_handle, class_name.encode(), attrs)
            self.native_op = native_op
            # for fast call
            native_op_handle = _ffi_api.GetNativeOpHandle(native_op)
            if isinstance(native_op_handle, ctypes.c_void_p):
                native_op_handle = native_op_handle.value
            self.native_op_handle = native_op_handle
        finally:
            _ffi_api.SetOpInitState(False)

    def __del__(self):
        if hasattr(self, "native_op"):
            self.__native_free_func(self.__backend_sess_handle, self.native_op)

    @property
    def name(self):
        if hasattr(self, "native_op"):
            return _ffi_api.OpHandleGetName(self.native_op)
        else:
            return ""

    @property
    def native_class_name(self):
        return self._native_class_name

    def __call__(self, *args, **kwargs):
        if len(kwargs) != 0:
            raise Exception("not support named args in this version")

        is_tracing = tracing()

        if is_tracing:
            args_data = []
            for arg in args:
                if isinstance(arg, BaseSymbol):
                    args_data.append(arg.data_2_71828182846)
                else:
                    args_data.append(arg)
            res_pack, type_code = op_kernel_call(self.native_op_handle, *args_data)
        else:
            res_pack, type_code = op_kernel_call(self.native_op_handle, *args)

        if not is_tracing:
            if type_code == TypeIndex.kRuntimeTuple:
                return tuple([x for x in res_pack])
            else:
                return res_pack

        # tracing
        converted_args = []  # don't delete this, used for hold symbol handle
        sym_handle_list = []
        for arg_idx, arg in enumerate(args):
            if isinstance(arg, OpKernel):
                # disable warning
                arg = Constant(arg)
            arg_conv, has_constant = make_symbol(arg, True)
            if has_constant:
                handle_trace_warning(_format_trace_warning_message(arg_idx))
            converted_args.append(arg_conv)
            sym_handle_list.append(arg_conv.native_handle_2_71828182846())
        num_output = 1
        sym_output_handles = _ffi_api.SymbolicExecutor_Compose(self.native_op,
                                                               num_output,
                                                               *sym_handle_list)
        sym_output = Symbol(sym_output_handles[0])
        sym_output.set_data_internal_2_71828182846(res_pack)
        return sym_output


def make_op_creator_function(op_class_name):
    assert isinstance(op_class_name, string_types)

    def creator(**kwargs):
        """Create a new op by kwargs

        Parameters
        ----------
        **kwargs :
            keyword arguments of this op

        Returns
        -------
        op : OpKernel


        """

        class Op(OpKernel):

            def __init__(self):
                super(Op, self).__init__(op_class_name, **kwargs)

            def __call__(self, *args, **kws):
                return super(Op, self).__call__(*args, **kws)

        Op.__name__ = op_class_name
        return Op()

    creator.__name__ = op_class_name
    return creator


class LibraryLoaderOp(OpKernel):

    def __init__(self, abi0_dl_paths, abi1_dl_paths):
        super().__init__("LibraryLoaderOp",
                         abi0_dl_paths=abi0_dl_paths,
                         abi1_dl_paths=abi1_dl_paths)

    def __call__(self, input_data):
        return super().__call__(input_data)


def make_library_loader_op(*args):
    abi0_dl_paths = []
    abi1_dl_paths = []
    for lib0, lib1 in args:
        abi0_dl_paths.append(lib0.encode())
        abi1_dl_paths.append(lib1.encode())
    return LibraryLoaderOp(abi0_dl_paths, abi1_dl_paths)


class DeviceOp(OpKernel):
    def __init__(self, device):
        super().__init__("DeviceOp",
                         device=device)

    def __call__(self):
        return super().__call__()


class InterpreterOp(OpKernel):
    def __init__(self, opcode,
                 py_source_file,
                 py_source_line,
                 py_source_func,
                 py_source_stmt):
        super().__init__("InterpreterOp",
                         opcode=opcode,
                         py_source_file=py_source_file,
                         py_source_line=py_source_line,
                         py_source_func=py_source_func,
                         py_source_stmt=py_source_stmt)

    def __call__(self, *args):
        return super().__call__(*args)
