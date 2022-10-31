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
from . import context
from . import utils
from . import analysis
from . import transforms
from .. import ir as _ir
from .parser import MATXScriptParser
from matx.env import MATX_DEV_MODE


def _passes(sc_ctx: context.ScriptContext):
    dep_anls = analysis.DepsAnalysis()
    src_anls = analysis.SourceAnalysis()
    mdo_anls = analysis.ModuleAnalysis()

    # parse main ast and do module analysis
    src_anls.run(sc_ctx)
    mdo_anls.run(sc_ctx)

    # alternate execution: parse deps, source and module analysis.
    while dep_anls.run(sc_ctx):
        src_anls.run(sc_ctx)
        mdo_anls.run(sc_ctx)

    # do renames before analysis
    name_trsf = transforms.NameNormalizer()
    rename_attrs_trans = transforms.RenameAttrsTransformer()
    if_bool_trans = transforms.IfBoolTransformer()
    name_trsf.run(sc_ctx)
    rename_attrs_trans.run(sc_ctx)
    if_bool_trans.run(sc_ctx)

    # function-and-class analysis
    fnc_anls = analysis.CallableAnalysis()
    fnc_anls.run(sc_ctx)

    # syntax check
    syntax_check = analysis.SyntaxCheck()
    inheri_consis_check = analysis.InheritencyConsistencyCheck()
    syntax_check.run(sc_ctx)
    inheri_consis_check.run(sc_ctx)

    # transform
    rename_call_super_trans = transforms.RenameCallSuperTransformer()
    rename_call_super_trans.run(sc_ctx)

    # build type
    analysis.BuildTypeAnalysis().run(sc_ctx)

    # other


def _parser(sc_ctx: context.ScriptContext):
    def parser_node(node: context.ASTNode):
        node.ir = MATXScriptParser(node, sc_ctx).visit(node.ast)
        # print(node.ir)

    for dep_node in sc_ctx.deps_node:
        parser_node(dep_node)
    parser_node(sc_ctx.main_node)


def _link_ir_module(sc_ctx: context.ScriptContext):
    ir_module = _ir.IRModule()

    def update_ir_module(name, func_or_mod):
        if isinstance(func_or_mod, _ir.BaseFunc):
            ir_module[name] = func_or_mod
        elif isinstance(func_or_mod, _ir.IRModule):
            ir_module.update(func_or_mod)
        else:
            raise TypeError('Only ir.BaseFunc and _ir.IRModule are supported.')

    update_ir_module(sc_ctx.main_node.context.name, sc_ctx.main_node.ir)
    for dep_node in sc_ctx.deps_node:
        update_ir_module(dep_node.context.name, dep_node.ir)
    sc_ctx.ir_module = ir_module


def _codegen(sc_ctx: context.ScriptContext):
    from .. import _ffi
    build_module = _ffi.get_global_func("module.build.c")
    if sc_ctx.build_type is context.BuildType.FUNCTION:
        fn_ctx: context.FunctionContext = sc_ctx.main_node.context
        sc_ctx.ir_module.add_export_func(fn_ctx.name)
    else:
        cls_ctx: context.ClassContext = sc_ctx.main_node.context
        for name, method in cls_ctx.methods.items():
            if name != '__init__':
                sc_ctx.ir_module.add_export_func(method.unbound_name)
    sc_ctx.rt_module = build_module(sc_ctx.ir_module)
    # print(sc_ctx.rt_module.get_source())


def _code_stat(sc_ctx: context.ScriptContext):
    from .code_statistics import MAIN_OBJ_STAT_INFO, COMPILING_OBJ_STAT_INFO
    main_node = sc_ctx.main_node
    main_node_name = main_node.raw.__name__
    compiling_objs = [main_node_name]
    codegen_source = sc_ctx.rt_module.get_source()
    MAIN_OBJ_STAT_INFO[main_node_name] = {"compiling_objs": compiling_objs,
                                          "codegen_co_lines": codegen_source.count('\n'),
                                          "codegen_co_chars": len(codegen_source)}
    COMPILING_OBJ_STAT_INFO[main_node_name] = {"co_lines": main_node.extra["co_lines"],
                                               "co_chars": main_node.extra["co_chars"],
                                               "is_class": main_node.extra["is_class"]}
    for dep_node in sc_ctx.deps_node:
        dep_node_name = dep_node.raw.__name__
        compiling_objs.append(dep_node_name)
        COMPILING_OBJ_STAT_INFO[dep_node_name] = {"co_lines": dep_node.extra["co_lines"],
                                                  "co_chars": dep_node.extra["co_chars"],
                                                  "is_class": main_node.extra["is_class"]}


def from_source(compiling_obj: type):
    try:
        sc_ctx = context.ScriptContext()
        sc_ctx.main_node.raw = compiling_obj
        _passes(sc_ctx)
        _parser(sc_ctx)
        _link_ir_module(sc_ctx)
        _codegen(sc_ctx)
        _code_stat(sc_ctx)
        return sc_ctx
    except BaseException as e:
        if MATX_DEV_MODE:
            raise
        else:
            raise Exception(str(e)) from None


def module(obj):
    sc_ctx = context.ScriptContext()
    sc_ctx.main_node.raw = obj
    _passes(sc_ctx)
    _parser(sc_ctx)
    _link_ir_module(sc_ctx)
    return sc_ctx.ir_module


def embedded_class_ctx(code):
    import inspect
    from .. import _ffi
    from .context import ClassContext, ScriptContext
    build_module = _ffi.get_global_func("embedded.build.c")
    sc_ctx = ScriptContext()
    if isinstance(code, str):
        code = code.encode()
    sc_ctx.rt_module = build_module(code)
    sc_ctx.main_node.context = ClassContext("Embedded")
    return sc_ctx
