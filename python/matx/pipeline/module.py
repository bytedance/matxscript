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
from __future__ import absolute_import as _abs
import os
import json
import sys
import warnings
from .._ffi.base import string_types
from .._ffi.error import trans_exception_from_c_to_py
from .._ffi import void_p_to_runtime
from . import _ffi_api
from .symbol import Variable
from .symbol import Symbol
from ._base import TXSession
import time


class Module(object):
    """
    Module Interface

    """

    def __init__(self, handle=None, name=None):
        self._tx_sess = TXSession(handle=handle, name=name)
        self.__holder = sys.modules['matx']

    def SetDevice(self, device):
        """Set devices used by Module tensor forward

        Parameters
        ----------
        device : int
            GPU serial numbers, or -1(CPU)

        Returns
        -------

        """
        assert isinstance(device, int), "devices is not int"
        return self._tx_sess.set_device(device)


class JITModule(Module):
    """Computing graph session management, responsible for the creation of resources, ops,
    variables, constants, etc. and their life cycle management

    """

    def __init__(self, handle=None, name=None):
        super(JITModule, self).__init__(handle, name)
        self._py_modules = []
        self._py_modules_trace_ret = []

    def _trace_py_module(self, sym_list):
        all_sym_deps = set()

        def _trace(_sym):
            if isinstance(_sym, Symbol):
                all_sym_deps.add(_sym)
                for _sym_i in _sym.get_inputs_2_71828182846():
                    _trace(_sym_i)

        for sym in sym_list:
            _trace(sym)

        self._py_modules = []
        for _sym in all_sym_deps:
            if _sym.py_instance_2_71828182846:
                self._py_modules.append(_sym.py_instance_2_71828182846)
        for py_m in self._py_modules:
            self._py_modules_trace_ret.append(py_m.trace())

    def _save_py_module(self):
        for i in range(len(self._py_modules)):
            self._py_modules[i].save(self._py_modules_trace_ret[i])

    def _save_code_stat_info(self, folder):
        import re
        from ..__init__ import __version__
        from ..__init__ import __commit_id__
        jit_op_instance_names, native_op_instance_names = _ffi_api.GetOpInstanceName(
            self._tx_sess.c_handle)
        jit_op_names = set()
        native_op_names = set()
        for jit_op_instance_name in jit_op_instance_names:
            m = re.search('^(.+?)_JitOp.*', jit_op_instance_name)
            if m:
                jit_op_names.add(m.group(1))
        for native_op_instance_name in native_op_instance_names:
            m = re.search('^(.+?)_[0-9]+', native_op_instance_name)
            if m:
                native_op_names.add(m.group(1))
        code_stat_info = {}
        code_stat_info["jit_op_count"] = len(jit_op_names)
        code_stat_info["native_op_count"] = len(native_op_names)
        from ..script.code_statistics import MAIN_OBJ_STAT_INFO, COMPILING_OBJ_STAT_INFO
        codegen_co_lines = 0
        codegen_co_chars = 0
        compiling_objs = set()
        for jit_op_name in jit_op_names:
            if jit_op_name not in MAIN_OBJ_STAT_INFO:
                print("[WARN] Can't find {} jitop statistics".format(jit_op_name))
                continue
            codegen_co_lines += MAIN_OBJ_STAT_INFO[jit_op_name]["codegen_co_lines"]
            codegen_co_chars += MAIN_OBJ_STAT_INFO[jit_op_name]["codegen_co_chars"]
            compiling_objs.update(MAIN_OBJ_STAT_INFO[jit_op_name]["compiling_objs"])
        python_class_count = 0
        python_function_count = 0
        python_co_lines = 0
        python_co_chars = 0
        for compiling_obj in compiling_objs:
            python_co_lines += COMPILING_OBJ_STAT_INFO[compiling_obj]["co_lines"]
            python_co_chars += COMPILING_OBJ_STAT_INFO[compiling_obj]["co_chars"]
            if COMPILING_OBJ_STAT_INFO[compiling_obj]["is_class"]:
                python_class_count += 1
            else:
                python_function_count += 1
        code_stat_info["python_class_count"] = python_class_count
        code_stat_info["python_function_count"] = python_function_count
        code_stat_info["python_co_lines"] = python_co_lines
        code_stat_info["python_co_chars"] = python_co_chars
        code_stat_info["codegen_co_lines"] = codegen_co_lines
        code_stat_info["codegen_co_chars"] = codegen_co_chars
        code_stat_info["version"] = __version__
        code_stat_info["commit_id"] = __commit_id__
        code_stat_info["create_time"] = int(time.time())

        with open(os.path.join(folder, "code_stat_info.json"), "w") as f:
            json.dump(code_stat_info, f, indent=4)

    def GetAttr(self, key):
        warnings.warn("The function JITModule.GetAttr is deprecated.", DeprecationWarning)
        return self.get_sess_attr(key)

    def get_sess_attr(self, key):
        return _ffi_api.TXSessionGetAttr(self._tx_sess.c_handle, key)

    def SetAttr(self, key, value):
        warnings.warn("The function JITModule.SetAttr is deprecated.", DeprecationWarning)
        return self.set_sess_attr(key, value)

    def set_sess_attr(self, key, value):
        _ffi_api.TXSessionSetAttr(self._tx_sess.c_handle, key, value)

    def HasAttr(self, key):
        warnings.warn("The function JITModule.HasAttr is deprecated.", DeprecationWarning)
        return self.has_sess_attr(key)

    def has_sess_attr(self, key):
        return _ffi_api.TXSessionHasAttr(self._tx_sess.c_handle, key)

    def InputNames(self):
        warnings.warn("The function JITModule.InputNames is deprecated.", DeprecationWarning)
        return self.input_names

    @property
    def input_names(self):
        return _ffi_api.TXSessionGetAttr(self._tx_sess.c_handle, "input_names")

    def SetInputNames(self, names):
        warnings.warn("The function JITModule.SetInputNames is deprecated.", DeprecationWarning)
        return _ffi_api.TXSessionSetAttr(self._tx_sess.c_handle, "input_names", names)

    @input_names.setter
    def input_names(self, names):
        return _ffi_api.TXSessionSetAttr(self._tx_sess.c_handle, "input_names", names)

    def SetOpParallelismThreads(self, thread_num=2, share=False):
        warnings.warn(
            "The function JITModule.SetOpParallelismThreads is deprecated.",
            DeprecationWarning)
        return self.set_op_parallelism_threads(thread_num, share)

    def set_op_parallelism_threads(self, thread_num=2, share=False):
        return self._tx_sess.set_op_parallelism_threads(thread_num=thread_num, share=share)

    def get_op_parallelism_threads(self):
        return self._tx_sess.get_op_parallelism_threads()

    def DisableOpParallelism(self):
        warnings.warn(
            "The function JITModule.DisableOpParallelism is deprecated.",
            DeprecationWarning)
        return self.disable_op_parallelism()

    def disable_op_parallelism(self):
        return self._tx_sess.disable_op_parallelism()

    def set_apply_async_threads(self, thread_num=2, share=False):
        return self._tx_sess.set_apply_async_threads(thread_num=thread_num, share=share)

    def get_apply_async_threads(self):
        return self._tx_sess.get_apply_async_threads()

    def disable_apply_async_threads(self):
        return self._tx_sess.disable_apply_async_threads()

    def set_pmap_threads(self, thread_num=8, share=False):
        return self._tx_sess.set_pmap_threads(thread_num=thread_num, share=share)

    def get_pmap_threads(self):
        return self._tx_sess.get_pmap_threads()

    def disable_pmap_threads(self):
        return self._tx_sess.disable_pmap_threads()

    def Trace(self, sym):
        warnings.warn("The function JITModule.Trace is deprecated.", DeprecationWarning)
        return self.trace(sym)

    def trace(self, sym):
        """Trace a function and return an executable module
        that will be optimized using just-in-time compilation.

        Parameters
        ----------
        sym : Symbol or list(Symbol)

        Returns
        -------
        """
        sym_list = []
        sym_handle_list = []
        if isinstance(sym, tuple) or isinstance(sym, list):
            for s in sym:
                if isinstance(s, (Symbol, Variable)):
                    sym_list.append(s)
                    sym_handle_list.append(s.native_handle_2_71828182846())
                else:
                    raise Exception("args is not BaseSymbol or BaseSymbol array")
        elif isinstance(sym, (Symbol, Variable)):
            sym_handle_list.append(sym.native_handle_2_71828182846())
            sym_list.append(sym)
        else:
            raise RuntimeError(
                f"Type {type(sym)} cannot be traced. Only the values returned by matx op can be traced. "
            )
        self._trace_py_module(sym_list)
        _ffi_api.TXSessionTrace(self._tx_sess.c_handle, *sym_handle_list)

    def Save(self, folder, name="model.spec.json"):
        warnings.warn("The function JITModule.Save is deprecated.", DeprecationWarning)
        return self.save(folder, name)

    def save(self, folder, name="model.spec.json"):
        """Save a Module to folder

        Parameters
        ----------
        folder : str
            model path

        name : str
            default, model.spec.json

        Returns
        -------

        """
        self._save_py_module()
        _ffi_api.TXSessionSave(self._tx_sess.c_handle, folder, name)
        self._save_code_stat_info(folder)

    def Run(self, feed_dict):
        warnings.warn("The function JITModule.Run is deprecated.", DeprecationWarning)
        return self.run(feed_dict)

    def __call__(self, *args, **kwargs):
        if len(args) != 0:
            # TODO: fixme
            raise TypeError("Only the kwargs is supported in this version")
        return self.run(kwargs)

    def run(self, feed_dict):
        """Execute Pipeline and get output

        Parameters
        ----------
        feed_dict : dict, matx.Dict
            The input feed dict

        Returns
        -------
        outputs : Tuple
            The output data

        """
        assert isinstance(feed_dict, dict), "feed_dict type error"
        feed_dict_v2 = dict()
        for k, v in feed_dict.items():
            k = k.encode()
            feed_dict_v2[k] = v
        fn_run = trans_exception_from_c_to_py(_ffi_api.TXSessionRun)
        try:
            result = fn_run(self._tx_sess.c_handle, feed_dict_v2)
            if len(result) == 1:
                return result[0]
            return tuple([obj for obj in result])
        except BaseException as e:
            e = type(e)(*e.args)
            raise e from None

    def warmup(self, feed_dict):
        """Warmup the Pipeline and get output

        Parameters
        ----------
        feed_dict : dict, matx.Dict
            The input feed dict

        Returns
        -------
        outputs : Tuple
            The output data

        """
        assert isinstance(feed_dict, dict), "feed_dict type error"
        feed_dict_v2 = dict()
        for k, v in feed_dict.items():
            k = k.encode()
            feed_dict_v2[k] = v
        result = _ffi_api.TXSessionWarmup(self._tx_sess.c_handle, feed_dict_v2)
        if len(result) == 1:
            return result[0]
        return tuple([obj for obj in result])

    def GenStepMeta(self, feed_dict):
        warnings.warn("The function JITModule.GenStepMeta is deprecated.", DeprecationWarning)
        return self.gen_step_meta(feed_dict)

    def gen_step_meta(self, feed_dict):
        """Execute Pipeline, then get step info

        Parameters
        ----------
        feed_dict : dict, matx.Dict
            The input feed dict

        Returns
        -------
        output : dict
            The step meta data

        """
        assert isinstance(feed_dict, dict), "feed_dict type error"
        feed_dict_v2 = dict()
        for k, v in feed_dict.items():
            k = k.encode()
            feed_dict_v2[k] = v
        result, meta = _ffi_api.TXSessionRunWithMeta(self._tx_sess.c_handle, feed_dict_v2)
        return meta

    def get_nested_op_attributes(self, op_cls, op_name):
        """Get all attributes of an op

        Parameters
        ----------
        op_cls: str, bytes
            Op Class Name

        op_name: str, bytes
            Op Instance Name

        Returns
        -------
        output : dict
            The Attributes

        """
        if isinstance(op_cls, str):
            op_cls = op_cls.encode()
        if isinstance(op_name, str):
            op_name = op_name.encode()
        assert isinstance(op_cls, (bytes, bytearray))
        assert isinstance(op_name, (bytes, bytearray))
        return _ffi_api.TXSessionGetNestedOpAttributesByName(
            self._tx_sess.c_handle, op_cls, op_name
        )

    def profile(self, feed_dict, warmup_times=10):
        """Execute Pipeline, get step info, generate timeline and show it

        Parameters
        ----------
        feed_dict : dict, matx.Dict
            The input feed dict

        warmup_times : int
            Times for warming up

        Returns
        -------
        """
        import prettytable
        print('warming up...')
        for i in range(warmup_times):
            self.run(feed_dict)
        meta = self.gen_step_meta(feed_dict)
        result_info = self.gen_timeline(meta)
        field_names = ["op", "start", "end", "time cost(ms)", "%"]
        output_tb = prettytable.PrettyTable()
        output_tb.field_names = field_names
        output_tb.add_rows(result_info)
        print(output_tb.get_string())

    @staticmethod
    def PrintTimeline(self, meta):
        warnings.warn("The function JITModule.PrintTimeline is deprecated.", DeprecationWarning)
        import prettytable
        result_info = self.gen_timeline(meta)
        field_names = ["op", "start", "end", "time cost(ms)", "%"]
        output_tb = prettytable.PrettyTable()
        output_tb.field_names = field_names
        output_tb.add_rows(result_info)
        print(output_tb.get_string())

    @staticmethod
    def gen_timeline(meta):
        import datetime

        def format_timestamp(ts):
            if ts == 0:
                return "-"
            m = (ts % 1000000000) / 1000000000.0
            ts = ts / 1000000000.0
            return datetime.datetime.utcfromtimestamp(ts).strftime('%H:%M:%S') + "%.6f" % m

        total_start = meta["start"]
        total_end = meta["end"]
        all_ts = (total_end - total_start) / 1000000.0
        result_info = []
        for op_info in meta["ops"]:
            start = op_info["start"]
            end = op_info["end"]
            op_ts = (end - start) / 1000000.0
            result_info.append(
                [
                    op_info["op"],
                    format_timestamp(start),
                    format_timestamp(end),
                    "%.4f(ms)" % op_ts,
                    "%.2f%%" % (op_ts * 100 / all_ts)
                ]
            )

        result_info.append(
            [
                "Total",
                format_timestamp(total_start),
                format_timestamp(total_end),
                "%.4f(ms)" % all_ts,
                "----"
            ]
        )
        return result_info


def LoadModule(folder, name, device):
    warnings.warn("The function matx.pipeline.LoadModule is deprecated.", DeprecationWarning)
    return load_module(folder, name, device)


def load_module(folder, name, device):
    """Load a matx model from folder

    Parameters
    ----------
    folder : str
        model path

    name : str
        config name

    device : int, str
        GPU serial numbers, or -1(CPU)

    Returns
    -------
    module : JITModule
        The executable module
    """
    from .. import runtime
    from ._plugin_loader import PluginLoader
    assert isinstance(folder, string_types)
    assert isinstance(name, string_types)
    if device is None:
        device = -1
    assert isinstance(device, (int, str))
    config = os.path.join(folder, name)
    with open(config, "r") as f:
        try:
            content = f.read()
        except:
            print("failed to read module config: ", config)
            raise
        root = runtime._ffi_api.pickle_FromJsonStruct(content)
        for op_obj in root[b"ops"]:
            op_class_name = op_obj[b"op"].decode()
            op_loader = PluginLoader.lookup(op_class_name)
            if op_loader:
                op_loader()
    handle = _ffi_api.LoadTXSession(folder, name, device)
    return JITModule(handle)
