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
from typing import List

import torch

from matx.torch_compiler.codegen import extract_inductor_code, matx_cpp_code_format
from .tensor_spec import TensorSpec
from .. import context, analysis
from ... import _ffi
from ... import ir
from ...env import MATX_DEV_MODE


def _embedded_inductor_ctx(compiling_obj, example_inputs):
    code = _obtain_inductor_code(compiling_obj, example_inputs)
    build_module = _ffi.get_global_func("embedded.build.c")
    sc_ctx = context.ScriptContext()
    sc_ctx.main_node.raw = compiling_obj
    if isinstance(code, str):
        code = code.encode()
    sc_ctx.rt_module = build_module(code)
    example_inputs_spec = [TensorSpec.from_tensor(inputs) for inputs in example_inputs]
    sc_ctx.main_node.context = context.InductorContext(fn_name=compiling_obj.__name__,
                                                       example_inputs_spec=example_inputs_spec)
    return sc_ctx


def _pass(sc_ctx: context.ScriptContext):
    src_anls = analysis.SourceAnalysis()
    src_anls.run(sc_ctx)


def _obtain_inductor_code(compiling_obj, example_inputs):
    # compile the kernel and set the code
    code, kernel_name, fake_output = extract_inductor_code(compiling_obj, example_inputs)
    code = matx_cpp_code_format(code, kernel_name, example_inputs, fake_output)
    return code


def from_source(compiling_obj: type, example_inputs: List[torch.Tensor]) -> context.ScriptContext:
    try:
        # TODO: allow generalized way to specify example_inputs
        sc_ctx = _embedded_inductor_ctx(compiling_obj, example_inputs)
        # set filename.
        _pass(sc_ctx)
        analysis.BuildTypeAnalysis().run(sc_ctx)

        # set args types.
        # TODO: currently, we only support argument as NDArray. We may support nested inputs later
        signature = inspect.signature(compiling_obj)
        for param in signature.parameters.values():
            sc_ctx.main_node.context.arg_types[param.name] = ir.type.NDArrayType()

        return sc_ctx
    except BaseException as e:
        if MATX_DEV_MODE:
            raise
        else:
            raise Exception(str(e)) from None
