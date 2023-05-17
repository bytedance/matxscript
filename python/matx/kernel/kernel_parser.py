#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import inspect

from matx.ir import _ffi_node_api
from matx.script import analysis
from matx.script import context as script_context
from .parser import KernelInspector
from .parser import extract_symbol_from_type


class KernelParser:

    def __init__(self, func):
        self.func = func
        self.func_name = func.__name__
        self.file_name = inspect.getfile(func)
        # get args
        self.signature = inspect.signature(func)
        self.args = {k: v.annotation for k, v in self.signature.parameters.items()}
        self.arg_types = list(self.args.values())
        # get return type
        self.return_types = self.signature.return_annotation
        # get shape symbols in dict like {'x':X}
        self.symbols = dict()
        for arg_type in self.arg_types:
            shape_symbol = extract_symbol_from_type(arg_type)
            self.symbols.update(shape_symbol)
        self.main_node_ir = None

    def passes(self, sc_ctx):
        dep_anls = analysis.DepsAnalysis()
        src_anls = analysis.SourceAnalysis()
        mdo_anls = analysis.ModuleAnalysis()
        src_anls.run(sc_ctx)
        mdo_anls.run(sc_ctx)

        while dep_anls.run(sc_ctx):
            src_anls.run(sc_ctx)
            mdo_anls.run(sc_ctx)

        fn_context = script_context.FunctionContext()
        # todo support default args
        fn_context.arg_defaults = []
        fn_context.arg_names = list(self.args.keys())
        # todo support args_reassigns
        fn_context.arg_reassigns = {k: False for k in self.args.keys()}
        # todo support types other than ndarray
        # todo fix type of fn_context and ir_schema
        fn_context.arg_types = self.args.items()
        fn_context.fn_name = self.func_name
        # context.fn_type = None
        fn_context.is_abstract = False
        fn_context.return_type = self.return_types
        fn_context.unbound_name = self.func_name
        sc_ctx.main_node.context = fn_context

        # sc_ctx.main_node.ir_schema = _ir.FuncType(
        #    list(fn_context.arg_types.values()), fn_context.return_type)

    def parse(self):
        sc_ctx = script_context.ScriptContext()
        sc_ctx.main_node.raw = self.func

        self.passes(sc_ctx)

        def parser_node(node: script_context.ASTNode):
            parser = KernelInspector(self, node).check_and_dispatch(node.ast)
            node.ir = parser.visit(node.ast)
            print(_ffi_node_api.IRTextPrinter_Print(node.ir, None).decode())
            return node.ir

        self.main_node_ir = parser_node(sc_ctx.main_node)

    def linalg_code(self):
        return _ffi_node_api.as_linalg_text(self.main_node_ir).decode()
