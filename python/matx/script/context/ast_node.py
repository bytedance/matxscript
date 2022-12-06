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
from typing import Callable, Optional, Tuple, Union, List, Dict, Any, Iterable, Set
from queue import Queue
from matx._typed_ast import ast
from .class_context import ClassContext
from .function_context import FunctionContext
from ... import ir as _ir


class Span:

    def __init__(self):
        self.file_name: str = ''
        self.lineno: int = 0
        self.func_name: str = ''
        self.source_code: str = ''


class ModuleInfo:

    def __init__(self):
        self.name: str = ''
        self.raw: Optional[type] = None
        self.globals: dict = {}
        self.imports: dict = {}


class ASTNode:

    def __init__(self, ):
        self.raw: Optional[type] = None
        self.span: Span = Span()
        self.ast: Optional[ast.AST] = None
        self.context: Union[ClassContext, FunctionContext, None] = None
        self.module: Optional[ModuleInfo] = None
        self.deps: Optional[List[ASTNode]] = None
        self.ir_schema = None
        self.ir = None
        self.last: Optional[ASTNode] = None
        self.extra: Dict[str, Any] = {}

    def __del__(self):
        # clear self.ir_schema
        if isinstance(self.ir_schema, _ir.ClassType):
            self.ir_schema.clear_members()

    def topodeps(self):
        outputs = []
        visited_raw = set()
        visited_deps = set()

        def dfs_visit(node: ASTNode):
            if node is None or node.raw is None:
                return
            if node.deps:
                if node.raw not in visited_deps:
                    # break ring
                    visited_deps.add(node.raw)
                    for dep in node.deps:
                        dfs_visit(dep)
            if node.raw not in visited_raw:
                outputs.append(node)
                visited_raw.add(node.raw)

        dfs_visit(self)
        return list(reversed(outputs))

    def get_dep_cls_by_name(self, name, deep=False):
        if self.context.name == name:
            return self
        if deep:
            for dep in self.deps:
                ret = dep.get_dep_cls_by_name(name)
                if ret:
                    return ret
        else:
            for dep in self.deps:
                if dep.context.name == name:
                    return dep
        return None

    def get_dep_cls_by_raw_type(self, raw_type, deep=False):
        if raw_type is self.context.raw:
            return self
        if deep:
            for dep in self.deps:
                ret = dep.get_dep_cls_by_raw_type(raw_type, deep)
                if ret:
                    return ret
        else:
            for dep in self.deps:
                if raw_type is dep.context.raw:
                    return dep
        return None

    def get_dep_by_module_attr(self, mod, attr):
        mod_attr = getattr(mod, attr)
        if self.raw is mod_attr:
            return self
        for dep in self.deps:
            if dep.raw is mod_attr:
                return dep
        return None

    def ir_call_wrapper(self, handle_var):
        from ... import ir as _ir
        from .. import rules

        err_msg_t = "could not convert the {}-th parameter type from '{}' to '{}'"

        if isinstance(self.context, ClassContext):
            init_func = rules.NameRule.get_class_init_wrapper(self.context.name)
            fn_ctx = self.context.init_fn

            def init_wrapper(span, *args, **kwargs):
                if self.context.is_abc and self.context.abstract_methods:
                    raise RuntimeError(
                        "Can't instantiate abstract class {} with abstract methods {}".format(
                            self.context.name, ', '.join(
                                self.context.abstract_methods)))
                sig = inspect.signature(self.raw)
                binds = sig.bind(*args, **kwargs)
                binds.apply_defaults()
                smart_args = []
                for i in range(len(binds.args)):
                    expect_ty = fn_ctx.arg_types[fn_ctx.arg_names[i]]
                    binds_arg_i = binds.args[i]
                    if not isinstance(binds_arg_i, _ir.BaseExpr):
                        if binds_arg_i is None:
                            binds_arg_i = _ir.NoneExpr()
                        else:
                            binds_arg_i = _ir.generic_const(binds_arg_i)
                    if not _ir.type_relation.type_convertible(
                            binds_arg_i.checked_type, expect_ty
                    ):
                        raise RuntimeError(
                            err_msg_t.format(
                                i + 1,
                                binds_arg_i.py_type_name(),
                                expect_ty.get_py_type_name()
                            )
                        )
                    smart_args.append(
                        _ir.type_relation.smart_adapt_to(binds_arg_i, expect_ty, span)
                    )
                smart_args.append(handle_var)
                return _ir.call_extern(self.ir_schema, init_func, span, *smart_args)

            init_wrapper.arg_types = fn_ctx.arg_types
            return init_wrapper
        elif isinstance(self.context, FunctionContext):
            fn_ctx = self.context

            def func_wrapper(span, *args, **kwargs):
                sig = inspect.signature(self.raw)
                binds = sig.bind(*args, **kwargs)
                binds.apply_defaults()
                smart_args = []
                for i in range(len(binds.args)):
                    expect_ty = fn_ctx.arg_types[fn_ctx.arg_names[i]]
                    binds_arg_i = binds.args[i]
                    if not isinstance(binds_arg_i, _ir.BaseExpr):
                        if binds_arg_i is None:
                            binds_arg_i = _ir.NoneExpr()
                        else:
                            binds_arg_i = _ir.generic_const(binds_arg_i)
                    if not _ir.type_relation.type_convertible(
                            binds_arg_i.checked_type, expect_ty
                    ):
                        raise RuntimeError(
                            err_msg_t.format(
                                i + 1,
                                binds_arg_i.py_type_name(),
                                expect_ty.get_py_type_name()
                            )
                        )
                    smart_args.append(
                        _ir.type_relation.smart_adapt_to(binds_arg_i, expect_ty, span)
                    )
                smart_args.append(handle_var)
                return _ir.call_extern(fn_ctx.return_type, fn_ctx.unbound_name, span, *smart_args)

            func_wrapper.arg_types = fn_ctx.arg_types
            return func_wrapper
        else:
            raise RuntimeError("unsupported context type: ", type(self.context))
