import inspect
from typing import List

import torch
from torch_compiler.manual_codegen import extract_inductor_code, matx_cpp_code_format

from matx.env import MATX_DEV_MODE
from matx.script import context
from matx.toolchain import path_prefix


def from_source(compiling_obj: type, example_inputs: List[torch.Tensor]) -> context.ScriptContext:
    try:

        code = extract_inductor_code(compiling_obj, example_inputs)
        code = matx_cpp_code_format(code)

        sc_ctx = context.ScriptContext()
        sc_ctx.build_type = context.BuildType.FUNCTION
        sc_ctx.main_node.raw = compiling_obj
        # set sc_ctx attributes to be compatible with existing matx code
        inductor_context = context.InductorContext(fn_name=compiling_obj.__name__)
        sc_ctx.main_node.context = inductor_context
        # set source code TODO: formatting source code
        sc_ctx.main_node.span.source_code = inspect.getsource(compiling_obj)
        # set filename. TODO: this is too hack
        frame = inspect.stack()[3]
        sc_ctx.main_node.span.file_name = frame[0].f_code.co_filename

        # export code
        path = path_prefix(sc_ctx)
        with open(path, 'w') as f:
            f.write(code)

        # set rt_module
        from .. import _ffi
        build_module = _ffi.get_global_func("embedded.build.c")
        sc_ctx.rt_module = build_module(code.encode())

        return sc_ctx
    except BaseException as e:
        if MATX_DEV_MODE:
            raise
        else:
            raise Exception(str(e)) from None
