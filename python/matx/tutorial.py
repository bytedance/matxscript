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

import ast
import astpretty
import inspect
from . import _ffi
from . import runtime
from . import ir as _ir
from .ir import generic as _generic
from .script import context
from typing import Any


class SimpleParser(ast.NodeVisitor):
    _op_maker = {
        ast.Add: lambda lhs, rhs: _generic.add(lhs, rhs),
    }
    _ty_maker = {
        "bool": lambda: _ir.PrimType("bool"),
        "int": lambda: _ir.PrimType("int64"),
        "float": lambda: _ir.PrimType("float64"),
    }

    def __init__(self):
        self.context = None
        self.functions = {}

    def init_function_parsing_env(self):
        """Initialize function parsing environment"""
        self.context = context.ScopeContext()

    def visit(self, node: ast.AST) -> Any:
        """Override method in ast.NodeVisitor"""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def parse_body(self):
        body = []
        while len(self.context.node_stack[-1]) > 0:
            res = self.visit(self.context.node_stack[-1].pop())
            if res is not None:
                body.append(res)
        if len(body) == 0:
            return _ir.SeqStmt(body)
        return _ir.SeqStmt(body) if len(body) > 1 else body[0]

    def generic_visit(self, node):
        """Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """
        raise RuntimeError('This node is not supported now: {}'.format(node), NotImplementedError)

    def visit_Module(self, node):
        """Module visitor"""

        if len(node.body) == 1 and isinstance(node.body[0], (ast.ClassDef, ast.FunctionDef)):
            # class or single function
            return self.visit(node.body[0])
        raise RuntimeError('Only one-function, one-class source code is allowed')

    def visit_FunctionDef(self, node: ast.FunctionDef):
        argtypes = []
        argnames = []
        self.init_function_parsing_env()
        self.context.new_scope(nodes=node.body)
        # add parameters of function
        for arg in node.args.args:
            var_type = self._ty_maker[arg.annotation.id]()
            argtypes.append(var_type)
            argnames.append(arg.arg)
            assert isinstance(var_type, _ir.PrimType)
            arg_var = _ir.PrimVar(arg.arg, var_type)
            self.context.update_symbol(arg.arg, arg_var)
            self.context.func_params.append(arg_var)
        ret_type = self._ty_maker[node.returns.id]()
        self.context.func_ret_type = ret_type

        # fetch the body and return a tir.PrimFunc
        func = _ir.Function(
            self.context.func_params,
            [],
            self.parse_body(),
            ret_type=ret_type,
        )
        func = func.with_attr("global_symbol", node.name)
        self.functions[node.name] = func

        self.context.pop_scope()
        return func

    def visit_Return(self, node):
        rt_expr = self.visit(node.value)
        return _ir.ReturnStmt(rt_expr)

    def lookup_or_alloca(self, name_hint, init_value):
        inf_ty = _ir.type_inference(init_value)
        symbol = self.context.lookup_symbol(name_hint)
        if symbol is None:
            alloca_stmt = _ir.AllocaVarStmt(name_hint, inf_ty, init_value)
            self.context.update_symbol(name_hint, alloca_stmt.var)
            return alloca_stmt
        else:
            return _ir.AssignStmt(symbol, init_value)

    def visit_Assign(self, node: ast.Assign):
        # https://docs.python.org/3/library/ast.html#ast.Assign
        if not len(node.targets) == 1:
            # a = b = c
            raise RuntimeError('Only one-valued assignment is supported now', NotImplementedError)
        lhs_node = node.targets[0]
        rhs_node = node.value
        rhs_value = self.visit(rhs_node)
        lhs_name = lhs_node.id
        return self.lookup_or_alloca(lhs_name, rhs_value)

    def visit_BinOp(self, node):
        """BinOp visitor
        AST abstract grammar:
            BinOp(expr left, operator op, expr right)
        """
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = self._op_maker[type(node.op)]
        return op(lhs, rhs)

    def visit_Name(self, node):
        name = node.id
        symbol = self.context.lookup_symbol(name)
        return symbol


def simple_compile(target, dso_path):
    func_name = target.__name__
    # get python source code
    source_code = inspect.getsource(target)

    # parse as ast
    ast_tree = ast.parse(source_code)
    print("python ast: ", flush=True)
    astpretty.pprint(ast_tree)

    # ast to ir
    parser = SimpleParser()
    func_ir = parser.visit(ast_tree)
    ir_module = _ir.IRModule()
    ir_module[func_name] = func_ir
    ir_module.set_main(func_name)

    print(ir_module)

    # codegen
    build_module = _ffi.get_global_func("module.build.c")
    rt_mod = build_module(ir_module)
    cc_code = rt_mod.get_source()
    base_options = [
        "-std=c++14",
        "-O3",
        "-g",
        "-fdiagnostics-color=always",
        "-Werror=return-type",
    ]
    cxx11_no_abi_options = base_options + ["-D_GLIBCXX_USE_CXX11_ABI=0"]
    rt_mod.export_library(dso_path, options=cxx11_no_abi_options, cc="g++")
    rt_mod = runtime.load_module(dso_path)
    compiled_func = rt_mod.get_function(func_name)
    return compiled_func, cc_code
