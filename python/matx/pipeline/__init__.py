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
import uuid
import inspect

from . import _register_conveter
from . import _ffi_api
from .._ffi.base import string_types
from .._ffi.error import trans_exception
from ..env import MATX_DEV_MODE
from ..contrib.statistic import counter
from .module import JITModule
from .module import Module
from .module import load_module, LoadModule
from . import ops
from . import warmup
from ._base import TXObject
from .ops import make_op_creator_function
from matx.pipeline.symbol import Symbol, Variable as Var
from matx.pipeline.symbol import Constant as Const
from .jit_object import JitOpImpl
from ._plugin_loader import PluginLoader
from ._tracing_state import begin_trace, finish_trace

# _ffi_api.SetPythonMode(True)


def make_module(func):
    """Experimental

    Parameters
    ----------
    func

    Returns
    -------

    """

    class MakeModule(Module):
        def __init__(self):
            super(MakeModule, self).__init__()

        def Execute(self, **kwargs):
            new_kwargs = dict()
            for k, v in kwargs.items():
                new_kwargs[k] = Var(k, v)
            return func(**new_kwargs)

    return MakeModule()


def method(func):
    """Experimental

    Parameters
    ----------
    func

    Returns
    -------

    """
    import inspect
    sig = inspect.signature(func)
    params = sig.parameters
    m = None
    if not params.get("self"):
        m = make_module(func)
    else:
        assert True

    def wrapper(*args, **kwargs):
        if m:
            return m.Execute(**kwargs)
        else:
            assert len(args) == 1
            new_kwargs = dict()
            for k, v in kwargs.items():
                new_kwargs[k] = Var(k, v)
            return func(*args, **new_kwargs)

    return wrapper


def TraceImpl(func, *args, **kwargs):
    """Trace a function and return an executable module that will be optimized using just-in-time compilation.

    Parameters
    ----------
    func : callable
        A Python function or a matx Symbol(s) that will be run with `args`.
        `func` arguments and return values must be Symbol.

    args :
        func inputs

    kwargs :
        func inputs

    Returns
    -------
    module : JITModule
       an executable module
    """
    if len(args) == 0 and len(kwargs) == 0 and not hasattr(func, '__call__'):
        jit_module = JITModule()
        jit_module.trace(func)
        return jit_module
    sig = inspect.signature(func)
    binds = sig.bind(*args, **kwargs)
    new_args = list()
    kwargs = {}
    for arg in binds.arguments.items():
        arg_name = arg[0]
        arg_value = arg[1]
        if arg_name == "kwargs":
            kwargs = arg_value
            continue
        if isinstance(arg_value, Var):
            new_args.append(arg_value)
        else:
            new_args.append(Var(arg_name, arg_value))
    new_kwargs = dict()
    for k, v in kwargs.items():
        if isinstance(v, Var):
            new_kwargs[k] = v
        else:
            new_kwargs[k] = Var(k, v)
    result = func(*new_args, **new_kwargs)
    jit_module = JITModule(name=str(uuid.uuid4()))
    jit_module.trace(result)
    jit_module.trace_result = result

    return jit_module


def trace(func, *args, **kwargs):
    begin_trace()
    try:
        mod = TraceImpl(func, *args, **kwargs)
    except Exception as e:
        # print(e)
        counter.set('matx_trace_error_counter', 1)
        if MATX_DEV_MODE:
            counter.set('matx_trace_error', str(e))
            counter.flush()
            raise
        else:
            ex, trans_state = trans_exception(e, use_cc_stacktrace=False)
            counter.set('matx_trace_error', str(ex))
            counter.flush()
            if trans_state:
                # raise from a jit object, keep same as origin
                raise
            else:
                raise ex from None
    finally:
        finish_trace()
    return mod


Trace = trace


def save(jit_module, folder, force_override=False):
    """Save a Module to folder

    Parameters
    ----------
    jit_module : JITModule
        The traced module

    folder : str
        the path be used to save model

    force_override : bool

    Returns
    -------

    """
    name = "model.spec.json"
    if (force_override
            and os.path.exists(folder)
            and folder.rstrip("/\\") not in ("/", ".", "..", "*")):
        print("rm old dir: %s" % folder)
        os.system("""rm -rf '%s' """ % folder)
    jit_module.save(folder, name)
    os.system("""chmod -R 755 '%s' """ % folder)
    os.system("""ls -l '%s' """ % folder)


Save = save


def SaveMeta(folder, model_name, allowed_batch_sizes):
    meta_file_name = "meta.pb.txt"
    assert os.path.exists(folder)
    full_pn = folder + os.sep + meta_file_name
    f = open(full_pn, "w")
    f.write('''name: "%s"\n\n''' % model_name)
    for batch_size in allowed_batch_sizes:
        f.write('''allowed_batch_sizes: %d\n''' % batch_size)
    f.close()
    os.system("""chmod -R 755 '%s' """ % full_pn)


def load(folder, device):
    """Load a matx model from folder

    Parameters
    ----------
    folder : str
        model path

    device : int, str
        GPU serial numbers, or -1(CPU)

    Returns
    -------
    module : JITModule
        The executable module

    """
    name = "model.spec.json"
    return load_module(os.path.abspath(folder), name, device)


Load = load


def _BuildSimpleGraph(op, signatures, num_output):
    converted_args = [Var(sig, None) for sig in signatures]
    sym_handle_list = [var.native_handle_2_71828182846() for var in converted_args]

    sym_output_handles = _ffi_api.SymbolicExecutor_Compose(
        op.native_op, num_output, *sym_handle_list)
    sym_outputs = [Symbol(x) for x in sym_output_handles]
    if len(sym_outputs) == 1:
        result = sym_outputs[0]
    else:
        result = tuple(sym_outputs)

    jit_module = JITModule(name=str(uuid.uuid4()))
    jit_module.trace(result)
    jit_module.trace_result = result
    jit_module.input_names = signatures
    return jit_module


def BuildSimpleGraph(op, signatures, num_output):
    begin_trace()
    try:
        module = _BuildSimpleGraph(op, signatures, num_output)
    finally:
        finish_trace()
    return module


_MATX_GLOBAL_PLUGINS_ = list()


def load_ops(lib_paths):
    """Load extend ops from dynamic lib and register to matx

    Parameters
    ----------
    lib_paths : str
        dynamic lib absolute path

    Returns
    -------
    """
    global _MATX_GLOBAL_PLUGINS_
    if isinstance(lib_paths, list):
        for pl in lib_paths:
            assert isinstance(pl, string_types)
    elif isinstance(lib_paths, string_types):
        lib_paths = [lib_paths]
    else:
        raise Exception(
            "expect library filepath or filepath list, but receive %s" % lib_paths.__class__)
    for op_dll in lib_paths:
        lib_pp = os.path.abspath(os.path.dirname(op_dll))
        cwd = os.getcwd()
        if lib_pp != "":
            os.chdir(lib_pp)
        lib = ctypes.CDLL(op_dll, ctypes.RTLD_GLOBAL)
        os.chdir(cwd)
        _MATX_GLOBAL_PLUGINS_.append(lib)
    for __op_cls_name in _ffi_api.ListAllOpNames():
        if getattr(ops, __op_cls_name, None) is None:
            __func = make_op_creator_function(__op_cls_name)
            setattr(ops, __op_cls_name, __func)
    return
