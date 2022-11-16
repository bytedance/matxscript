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
import sys
from matx import pipeline


def check(compiling_obj):
    torch = sys.modules.get('torch')
    if torch is not None and hasattr(torch, 'nn') and hasattr(torch, 'jit'):
        if inspect.isclass(compiling_obj) and issubclass(compiling_obj, torch.nn.Module):
            return True
        return isinstance(compiling_obj, (torch.jit.ScriptModule, torch.nn.Module))
    return False


def _torch_script_module_trace(mod, *args, **kwargs):
    return mod


def script(compiling_obj, *, trace_func=None, **kwargs):
    import torch
    from .pytorch_module import PytorchModule, make_pipeline_op_from_location
    if inspect.isclass(compiling_obj) and issubclass(compiling_obj, torch.nn.Module):
        raise NotImplementedError('type of torch.nn.Module')
        # from .pytorch_module import TorchModuleMixin
        # timestamp = int(round(time.time() * 1000))
        # new_obj = type(f'TorchModuleMixin_{timestamp}', (TorchModuleMixin, compiling_obj), {})
        # new_obj._trace_func = trace_func
        # new_obj._script_args = args
        # new_obj._script_kwargs = kwargs
        # return new_obj
    else:
        if isinstance(compiling_obj, str):
            return make_pipeline_op_from_location(location=compiling_obj, **kwargs)
        if isinstance(compiling_obj, (torch.jit.ScriptModule, torch.jit.ScriptFunction)):
            if trace_func is None:
                trace_func = _torch_script_module_trace
        mod = PytorchModule(model=compiling_obj, trace_func=trace_func, **kwargs)
        return mod.make_pipeline_op()


def _compile_or_load_lib_wrapper():
    from .lib import compile_or_load_lib
    compile_or_load_lib()


pipeline.PluginLoader.register("PyTorchInferOp", _compile_or_load_lib_wrapper)
pipeline.PluginLoader.register("TorchInferOp", _compile_or_load_lib_wrapper)
pipeline.PluginLoader.register("TorchModel", _compile_or_load_lib_wrapper)
