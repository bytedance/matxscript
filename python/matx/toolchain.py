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
import logging
import hashlib
import inspect

from typing import Dict, List, Any, Optional
from ._ffi.base import _LIB_SHA1
from . import contrib
from .script import from_source
from .script import context
from .script import embedded_class_ctx
from .pipeline.ops import OpKernel
from .pipeline.jit_object import JitObject
from .pipeline.jit_object import JitOpImpl
from . import runtime
from ._ffi.libinfo import find_include_path

LIB_PATH = os.environ.get('MATX_DSO_DIR', 'dso')

USE_SO_CACHE = os.environ.get('MATX_USE_SO_CACHE', '').lower() != 'false'

DISABLE_SCRIPT = os.environ.get('MATX_DISABLE_SCRIPT', '').lower() == 'true'
DISABLE_GENERATE_CC = os.environ.get('MATX_DISABLE_GENERATE_CC', '').lower() == 'true'
FLAG_COMPILED_OBJECT = object()


class ArgumentValueError(ValueError):
    pass


def _mk_lib_dir():
    if os.path.exists(LIB_PATH) and not os.path.isdir(LIB_PATH):
        raise RuntimeError(
            'Default compiling result path dso is not a directory.')

    os.makedirs(LIB_PATH, exist_ok=True)


class PathGuesser:
    def __init__(self) -> None:
        self.need_bundle = []
        self.jsonpath = '$'

    def guess(self, params: Any) -> None:
        if isinstance(params, Dict):
            for k, v in params.items():
                jsonpath = self.jsonpath
                self.jsonpath += '.{}'.format(k)
                self.guess(v)
                self.jsonpath = jsonpath
        elif isinstance(params, List):
            for i, v in enumerate(params):
                jsonpath = self.jsonpath
                self.jsonpath += '[{}]'.format(i)
                self.guess(v)
                self.jsonpath = jsonpath
        elif isinstance(params, str):
            if not set(params).difference(['.', '/']):
                pass
            elif os.path.exists(params):
                self.need_bundle.append(self.jsonpath)
        else:
            pass


class ToolChain:
    def __init__(self, toolchain=None, **kwargs):
        self.toolchain = toolchain
        self.kwargs = kwargs

    def build(self, file_path):
        if self.toolchain == "ndk":
            from .contrib import ndk as _ndk
            _ndk.create_shared(file_path, find_include_path(), **self.kwargs)
        else:
            raise RuntimeError("unsupport toolchain: %s" % self.toolchain)

    def __str__(self):
        return "toolchain: {}, kwargs: {}".format(
            self.toolchain, self.kwargs)


def make_jit_op_creator(sc_ctx: context.ScriptContext, share=True, bundle_args=None):
    assert sc_ctx.build_type == context.BuildType.FUNCTION
    main_func_name = sc_ctx.main_node.context.name
    jit_obj_creator = make_jit_object_creator(sc_ctx, share, bundle_args=bundle_args)

    def jit_op_creator(*args, **kwargs):
        r = jit_obj_creator(*args, **kwargs)
        jit_op_imp = JitOpImpl(main_func_name=main_func_name, jit_object=r)
        jit_op_imp.__name__ = sc_ctx.main_node.context.name
        return jit_op_imp

    jit_op_creator.__FLAG_COMPILED_OBJECT__ = FLAG_COMPILED_OBJECT
    return jit_op_creator


def make_jit_object_creator(sc_ctx: context.ScriptContext, share=True, bundle_args=None):
    from .pipeline.jit_object import FuncMeta, FuncParam, ClassMeta
    from .pipeline.jit_object import restore_user_behavior
    is_function = sc_ctx.build_type == context.BuildType.FUNCTION
    user_class_name = sc_ctx.main_node.context.name

    captures = []
    if sc_ctx.free_vars:
        for var_ins in sc_ctx.free_vars:
            assert isinstance(var_ins, OpKernel)
            captures.append((var_ins.native_class_name.encode(), var_ins.name.encode()))

    def make_func_meta(func_ctx, bound_self, params=None, self_type=None):
        func_args = []
        init_args = []
        if bound_self:
            func_args.append(FuncParam("self", self_type.get_runtime_type_code()))
        for arg_name, arg_type in func_ctx.arg_types.items():
            arg_type = arg_type.get_runtime_type_code()
            if params is not None:
                try:
                    init_args_i = runtime.to_runtime_object(params[arg_name])
                    init_args.append(init_args_i)
                except ValueError as e:
                    msg = f"The value of argument '{arg_name}' is invalid, "
                    raise ArgumentValueError(msg + e.args[0]) from None
            func_args.append(FuncParam(arg_name, arg_type))
        arg_defaults = []
        func_meta = FuncMeta(func_ctx.unbound_name, bound_self, func_args, arg_defaults)
        if params is None:
            return func_meta
        else:
            return func_meta, init_args

    def jit_object_creator(*args, **kwargs):
        if is_function:
            fn_ctx = sc_ctx.main_node.context
            func_name = fn_ctx.name
            func_schema = make_func_meta(fn_ctx, bound_self=False)
            ud = JitObject(
                dso_path=sc_ctx.dso_path[0],
                dso_path_cxx11=sc_ctx.dso_path[1],
                meta_info=func_schema,
                function_mapping={func_name: func_name},
                share=share,
                captures=captures,
                py_source_file=sc_ctx.main_node.span.file_name.encode(),
                py_source_line=sc_ctx.main_node.span.lineno,
            )
            ud = restore_user_behavior(ud, func_name, False, func_schema)
            ud.__name__ = sc_ctx.main_node.context.name
            return ud
        else:
            cls_ctx = sc_ctx.main_node.context
            self_type = sc_ctx.main_node.ir_schema
            try:
                # None for self
                init_func_signature = inspect.signature(sc_ctx.main_node.raw.__init__)
                init_func_params = init_func_signature.bind(
                    None, *args, **kwargs)
                init_func_params.apply_defaults()
                init_func_params = init_func_params.arguments
            except TypeError as e:
                e.args = (e.args[0] + ' when calling __init__ of ' +
                          user_class_name,)
                raise
            guesser = PathGuesser()
            if bundle_args is None:
                guesser.guess(init_func_params)
            else:
                if not isinstance(bundle_args, (list, set, tuple)):
                    raise TypeError("matx.script: bundle_args must be None or List")
                guess_arguments = type(init_func_params)()
                for init_func_param_name, init_func_param_value in init_func_params.items():
                    if init_func_param_name in bundle_args:
                        guess_arguments[init_func_param_name] = init_func_param_value
                guesser.guess(guess_arguments)
            need_bundle = guesser.need_bundle
            logging.info('need_bundle: %s', need_bundle)
            try:
                ctor_func_meta, init_args = make_func_meta(cls_ctx.init_fn,
                                                           bound_self=False,
                                                           params=init_func_params)
            except ArgumentValueError as e:
                raise ValueError(*e.args) from None
            member_funcs = [ctor_func_meta]
            for fn_name, fn_ctx in cls_ctx.methods.items():
                bound_self = fn_ctx.fn_type == context.FunctionType.INSTANCE
                member_funcs.append(make_func_meta(fn_ctx, bound_self, self_type=self_type))

            class_info = ClassMeta(name=user_class_name,
                                   len_slots=len(cls_ctx.attr_names),
                                   init_func=ctor_func_meta,
                                   init_args=init_args,
                                   member_funcs=member_funcs)
            function_mapping = {
                fn_name: fn_ctx.unbound_name for fn_name, fn_ctx in cls_ctx.methods.items()
            }
            function_mapping[cls_ctx.init_fn.name] = cls_ctx.init_fn.unbound_name
            ud = JitObject(
                dso_path=sc_ctx.dso_path[0],
                dso_path_cxx11=sc_ctx.dso_path[1],
                meta_info=class_info,
                need_bundle=need_bundle,
                function_mapping=function_mapping,
                share=share,
                captures=captures,
                py_source_file=sc_ctx.main_node.span.file_name.encode(),
                py_source_line=sc_ctx.main_node.span.lineno,
            )
            ud = restore_user_behavior(ud, user_class_name, True, ctor_func_meta, member_funcs)
            ud.__name__ = sc_ctx.main_node.context.name
            return ud

    jit_object_creator.__name__ = user_class_name
    jit_object_creator.__RAW_TYPE_2_71828182846___ = sc_ctx.main_node.raw
    jit_object_creator.__FLAG_COMPILED_OBJECT__ = FLAG_COMPILED_OBJECT
    return jit_object_creator


def path_prefix(sc_ctx: context.ScriptContext):
    # mkdir LIB_PATH
    from .__init__ import __version__
    _mk_lib_dir()
    # code + sha1(libmatx.so) + commit_id(__version__)
    dep_source_codes = "".join(dep_node.span.source_code for dep_node in sc_ctx.deps_node)
    cache_str = sc_ctx.main_node.span.source_code + dep_source_codes + _LIB_SHA1 + __version__
    cache_md5 = hashlib.md5(cache_str.encode()).hexdigest()[:16]
    file_name = os.path.splitext(os.path.basename(sc_ctx.main_node.span.file_name))[0]
    return os.path.abspath('{}/lib{}_{}_{}_plugin_{}'.format(LIB_PATH,
                                                             file_name,
                                                             sc_ctx.main_node.span.lineno,
                                                             sc_ctx.main_node.context.name,
                                                             cache_md5))


def toolchain_path_prefix(sc_ctx: context.ScriptContext, toolchain_str: str):
    from .__init__ import __version__
    # mkdir LIB_PATH
    _mk_lib_dir()
    # code + sha1(libmatx.so) + commit_id(__version__) + toolchain_str
    dep_source_codes = "".join(dep_node.span.source_code for dep_node in sc_ctx.deps_node)
    cache_str = sc_ctx.main_node.span.source_code + dep_source_codes
    cache_str += _LIB_SHA1 + __version__ + toolchain_str
    cache_md5 = hashlib.md5(cache_str.encode()).hexdigest()[:16]
    file_name = os.path.splitext(os.path.basename(sc_ctx.main_node.span.file_name))[0]
    return os.path.abspath('{}/lib{}_{}_{}_plugin_{}'.format(LIB_PATH,
                                                             file_name,
                                                             sc_ctx.main_node.span.lineno,
                                                             sc_ctx.main_node.context.name,
                                                             cache_md5))


def hit_cache(so_path: str):
    if USE_SO_CACHE and os.path.isfile(so_path):
        return True
    return False


def toolchain_build(sc_ctx: context.ScriptContext, toolchain: ToolChain):
    rt_mod = sc_ctx.rt_module
    main_node_name = sc_ctx.main_node.context.name
    base_path = toolchain_path_prefix(sc_ctx, str(toolchain))

    with contrib.util.filelock(base_path):
        file_path = base_path + '_android.cc'
        so_path = base_path + '_android.so'
        # save file
        rt_mod.save(file_path)
        if not hit_cache(so_path):
            logging.info(
                "matx compile function/class: [{}:{}]".format(main_node_name, so_path))
            toolchain.build(file_path)
        else:
            logging.info(
                "info matched, skip compiling: [{}:{}]".format(
                    main_node_name, so_path))

        sc_ctx.dso_path = (sc_ctx.dso_path[0], so_path)


def build_dso(sc_ctx: context.ScriptContext, use_toolchain=False):
    rt_mod = sc_ctx.rt_module
    main_node_name = sc_ctx.main_node.context.name
    base_path = path_prefix(sc_ctx)

    with contrib.util.filelock(base_path):
        sopath = base_path + '.so'
        sopath_cxx11 = base_path + '_cxx11.so'

        base_options = [
            "-std=c++14",
            "-O3",
            "-g",
            "-fdiagnostics-color=always",
            "-Werror=return-type"]
        cxx11_with_abi_options = base_options + ["-D_GLIBCXX_USE_CXX11_ABI=1"]
        cxx11_no_abi_options = base_options + ["-D_GLIBCXX_USE_CXX11_ABI=0"]
        sys_cc_path = contrib.cc.find_sys_cc_path()

        contrib.cc.check_cc_version(sys_cc_path, False)
        if not hit_cache(sopath):
            logging.info("matx compile function/class: [{}:{}]".format(main_node_name, sopath))
            rt_mod.export_library(sopath, options=cxx11_no_abi_options, cc=sys_cc_path)
        else:
            logging.info("info matched, skip compiling: [{}:{}]".format(main_node_name, sopath))
        if not use_toolchain:
            server_cc_path = contrib.cc.find_server_gcc_path()
            if server_cc_path is not None:
                contrib.cc.check_cc_version(server_cc_path, True)
                if not hit_cache(sopath_cxx11):
                    logging.info(
                        "matx compile function/class: [{}:{}]".format(main_node_name, sopath_cxx11))
                    rt_mod.export_library(
                        sopath_cxx11,
                        options=cxx11_with_abi_options,
                        cc=server_cc_path)
                else:
                    logging.info(
                        "info matched, skip compiling: [{}:{}]".format(
                            main_node_name, sopath_cxx11))
            else:
                msg_t = "matx compile \"{}\": server gcc not found, will disable build \"{}\""
                logging.warning(msg_t.format(main_node_name, os.path.basename(sopath_cxx11)))
                sopath_cxx11 = ""
        else:
            sopath_cxx11 = ""

        sc_ctx.dso_path = (sopath, sopath_cxx11)


def disable_script():
    global DISABLE_SCRIPT
    DISABLE_SCRIPT = True


def script(compiling_obj, *, share=True, toolchain=None, bundle_args=None):
    """Entry function for compiling. Given a python object including function,
    simple class, compile it to a matx4 object which mostly
    keep the behavior of the original python object.

    Args:
        compiling_obj ([function, class]): [input python object to be compiled.]
        share (bool): if share this object
        toolchain (class): custom toolchains used to compile the generated c++ files
        bundle_args (list of str):

    Returns:
        the compiled object.
    """
    if DISABLE_SCRIPT:
        return compiling_obj
    result: context.ScriptContext = from_source(compiling_obj)
    build_dso(result, toolchain is not None)
    if toolchain is not None:
        toolchain_build(result, toolchain)

    if result.build_type is context.BuildType.FUNCTION:
        return make_jit_op_creator(result, share, bundle_args=bundle_args)()
    elif result.build_type is context.BuildType.JIT_OBJECT:
        return make_jit_object_creator(result, share, bundle_args=bundle_args)
    else:
        raise ValueError('Unsupported build_type: {}'.format(result.build_type))


def make_session(compiling_obj, method='__call__'):
    from . import pipeline

    def _has_method(cls):
        for name, _ in inspect.getmembers(cls, inspect.isfunction):
            if name == method:
                return True
        return False

    build_type = None
    sig_obj = compiling_obj
    if inspect.isclass(compiling_obj):
        assert _has_method(compiling_obj)
        build_type = context.BuildType.JIT_OBJECT
        sig = inspect.signature(getattr(sig_obj, method))
        if method == '__call__':
            method = 'native_call_method'

    elif inspect.isfunction(compiling_obj):
        build_type = context.BuildType.FUNCTION
        sig = inspect.signature(sig_obj)
    else:
        raise RuntimeError("Only functions and classes are scriptable.")

    signature = [param[0] for param in sig.parameters]
    scripted_obj = script(compiling_obj)

    if build_type == context.BuildType.FUNCTION:
        return pipeline.BuildSimpleGraph(scripted_obj, signature, 1)
    elif build_type == context.BuildType.JIT_OBJECT:
        def _session_creator(*args, **kwargs):
            jit_obj = scripted_obj(*args, **kwargs)
            # retrieve the JitOpImpl instance for this method
            op = jit_obj.op_mapping_2_71828182846[method]
            return pipeline.BuildSimpleGraph(op, signature[1:], 1)

        return _session_creator
    else:
        # Unreachable code branch.
        return None


def script_embedded_class(code, is_path=False):
    if is_path:
        code = open(code).read()
    sc_ctx = embedded_class_ctx(code)
    previous_frame = inspect.currentframe().f_back
    frame = inspect.getframeinfo(previous_frame)
    sc_ctx.main_node.span.file_name = os.path.abspath(frame.filename)
    sc_ctx.main_node.span.lineno = frame.lineno
    sc_ctx.main_node.span.code = code
    build_dso(sc_ctx)
    return sc_ctx.dso_path
