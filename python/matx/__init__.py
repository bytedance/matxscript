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
# pylint: disable=redefined-builtin, wildcard-import
from . import _hooks
from . import runtime
from .contrib import cpp_extension
from .contrib.cpp_extension import get_cflags, get_link_flags
from . import toolchain
from . import pipeline
from .toolchain import ToolChain
from . import extension
from .runtime import msgpack_loads, msgpack_dumps
from . import text

try:
    from . import vision
except RuntimeError as e:
    import sys
    print(e, file=sys.stderr)
    print("vision sub modules requires manually downloading shared lib to matxscript/vision", file=sys.stderr)
    vision = None


# APIs
__all__ = [
    # functions
    "list_sort",
    "pmap",
    "pstarmap",
    "load_so",
    "trace",
    "script",
    "script_embedded_class",
    "save",
    "load",
    "get_cflags",
    "get_link_flags",

    # alias
    "runtime",
    "toolchain",
    "pipeline",
    "cpp_extension",
    "ToolChain",
    "extension",

    # version info
    "__version__",
    "__branch__",
    "__commit_id__",

    # matx._ffi
    "TError",
    "register_func",
    "get_global_func",
    "get_global_func",

    # matx.runtime
    "Object",
    "NDArray",
    "array",
    "List",
    "Dict",
    "Set",
    "Tuple",
    "OpaqueObject",
    "to_runtime_object",
    "File",
    "Regex",
    "Trie",
    "list_heapify",
    "list_heap_replace",
    "list_nth_element",
    "list_heap_pushpop",
    "serialize",
    "deserialize",

    # matx.native
    "native",
    "NativeObject",
    "make_native_object_creator",
    "make_native_object",
    "make_native_function",
    "load_native",

    # modules
    "pypi",
    "ir",
    "ir_module",
    "contrib",
    "error",
    "Device",
]

# version info
__version__ = "1.8.0.alpha"
__branch__ = None
__commit_id__ = None

# top-level alias
# matx._ffi
from ._ffi.base import TError
from ._ffi import register_func, get_global_func, to_packed_func

# top-level alias
# matx.runtime
from .runtime.object import Object
from .runtime.ndarray import NDArray
from .runtime import ndarray as array
from .runtime import List, Dict, Set, Tuple
from .runtime import OpaqueObject
from .runtime.object_generic import to_runtime_object
from .runtime.file import File
from .runtime.regex import Regex
from .runtime.trie import Trie
from .runtime._container._list import heapify as list_heapify
from .runtime._container._list import heap_replace as list_heap_replace
from .runtime._container._list import nth_element as list_nth_element
from .runtime._container._list import heap_pushpop as list_heap_pushpop

from .runtime.picke import serialize, deserialize

# matx.native
from . import native
from .native import NativeObject
from .native import make_native_object_creator
from .native import make_native_object
from .native import make_native_function
from .native import call_native_function
from .native import load_native

# matx.pypi
from . import pypi

# matx.ir
from . import ir

from .script import module as ir_module

# matx.contrib
from . import contrib

# matx.error
from . import error

from .typing import *

from .pipeline.ops import DeviceOp as Device


# compiling api


def list_sort(l, compare=None):
    if compare is None:
        if isinstance(l, list):
            l.sort()
        elif isinstance(l, List):
            from .runtime import _ffi_api
            _ffi_api.ListSort(l)
        else:
            raise ValueError("type of first arg must be list")
    else:
        from functools import cmp_to_key
        if isinstance(l, list):
            l.sort(key=cmp_to_key(compare))
        elif isinstance(l, List):
            from .pipeline.ops import OpKernel
            if not isinstance(compare, OpKernel):
                tmp = sorted(l, key=cmp_to_key(compare))
                l.clear()
                for item in tmp:
                    l.append(item)
            else:
                from .runtime import _ffi_api
                _ffi_api.ListSort(l, compare)
        else:
            raise ValueError("type of first arg must be list")


def pmap(func, data):
    from . import pipeline
    from .pipeline._base import TXObject
    from .pipeline import _ffi_api
    from .pipeline._tracing_state import tracing

    if tracing():
        from .pipeline import builtin_op
        pmap_op = builtin_op.get_interpreter_op("ParallelMap")
        if not isinstance(func, (pipeline.ops.OpKernel, runtime.object.ObjectBase)):
            raise TypeError(f"matx.pmap: the first argument '{func}' is not a traceable op")
        if not isinstance(
                data, (pipeline.symbol.BaseSymbol, runtime.List, runtime.Tuple, list, tuple)
        ):
            raise TypeError(f"matx.pmap: the second argument '{data}' is not supported")
        return pmap_op(func, data)
    if not isinstance(func, (pipeline.ops.OpKernel, runtime.object.ObjectBase)):
        # Python mode
        if isinstance(data, (list, runtime.List)):
            return [func(x) for x in data]
        elif isinstance(data, (tuple, runtime.Tuple)):
            return tuple(func(x) for x in data)
        else:
            raise TypeError(f"expect the second argument is list or tuple, but get '{data}'")
    sess_handle = TXObject.default_sess.c_handle
    return _ffi_api.ParallelMap(func, data, sess_handle)


def pstarmap(func, data):
    from . import pipeline
    from .pipeline._base import TXObject
    from .pipeline import _ffi_api
    from .pipeline._tracing_state import tracing

    if tracing():
        from .pipeline import builtin_op
        pstarmap_op = builtin_op.get_interpreter_op("ParallelStarMap")
        if not isinstance(func, (pipeline.ops.OpKernel, runtime.object.ObjectBase)):
            raise TypeError(f"matx.pstarmap: the first argument '{func}' is not a traceable op")
        if not isinstance(
                data, (pipeline.symbol.BaseSymbol, runtime.List, runtime.Tuple, list, tuple)
        ):
            raise TypeError(f"matx.pstarmap: the second argument '{data}' is not supported")
        return pstarmap_op(func, data)
    if not isinstance(func, (pipeline.ops.OpKernel, runtime.object.ObjectBase)):
        # Python mode
        if isinstance(data, (list, runtime.List)):
            return [func(*x) for x in data]
        elif isinstance(data, (tuple, runtime.Tuple)):
            return tuple(func(*x) for x in data)
        else:
            raise TypeError(f"expect the second argument is list or tuple, but get '{data}'")
    sess_handle = TXObject.default_sess.c_handle
    return _ffi_api.ParallelStarMap(func, data, sess_handle)


class Future:
    def __init__(self, x):
        self.__x = x

    def get(self):
        return self.__x

    def __call__(self, ):
        return self.__x


def apply_async(func, *args):
    from . import pipeline
    from .pipeline._base import TXObject
    from .pipeline import _ffi_api
    from .pipeline._tracing_state import tracing

    if tracing():
        from .pipeline import builtin_op
        async_op = builtin_op.get_interpreter_op("ApplyAsync")
        if not isinstance(func, (pipeline.ops.OpKernel, runtime.object.ObjectBase)):
            raise TypeError(f"matx.pmap: the first argument '{func}' is not a traceable op")
        return async_op(func, *args)
    if not isinstance(func, (pipeline.ops.OpKernel, runtime.object.ObjectBase)):
        # Python mode
        return Future(func(*args))
    sess_handle = TXObject.default_sess.c_handle
    return _ffi_api.ApplyAsync(func, *args, sess_handle)


def load_so(dso_path=""):
    from ._ffi.base import USE_CXX11_ABI
    if isinstance(dso_path, str):
        load_native(dso_path)
    elif isinstance(dso_path, tuple):
        if USE_CXX11_ABI():
            load_native(dso_path[1])
        else:
            load_native(dso_path[0])


def trace(func, *args, **kwargs):
    try:
        return pipeline.trace(func, *args, **kwargs)
    except BaseException:
        raise


def script(compiling_obj, *args, backend=None, **kwargs):
    from . import extension
    if isinstance(compiling_obj, pipeline.ops.OpKernel):
        return compiling_obj
    if getattr(compiling_obj, "__FLAG_COMPILED_OBJECT__", None) is toolchain.FLAG_COMPILED_OBJECT:
        return compiling_obj
    if isinstance(backend, str):
        backend = backend.lower()
    if backend in ('torch', 'pytorch'):
        return extension.pytorch.script(compiling_obj, *args, **kwargs)
    elif backend in ('tensorflow',):
        return extension.tensorflow.script(compiling_obj, *args, **kwargs)
    else:
        if extension.pytorch.check(compiling_obj):
            return extension.pytorch.script(compiling_obj, *args, **kwargs)
        elif extension.tensorflow.check(compiling_obj):
            return extension.tensorflow.script(compiling_obj, *args, **kwargs)
        return toolchain.script(compiling_obj, *args, **kwargs)


def script_embedded_class(code, is_path=False):
    return toolchain.script_embedded_class(code, is_path)


def save(jit_module, folder, force_override=False):
    return pipeline.save(jit_module, folder, force_override)


def load(folder, device):
    return pipeline.load(folder, device)
