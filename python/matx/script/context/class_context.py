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
from typing import Dict, Optional, List, Set

from .function_context import FunctionContext, FunctionType
from .typing import AnnotatedType
from ... import ir as _ir


class GetClassAttr:

    def __init__(self,
                 cls_name,
                 attr_name,
                 func_ctx,
                 symbol,
                 call_func,
                 unbound_func_name,
                 session_pointer_var):
        super().__init__()
        self.cls_name = cls_name
        self.attr_name = attr_name
        self.arg_types = func_ctx.arg_types
        self.symbol = symbol
        self.call_func = call_func
        self.session_pointer_var = session_pointer_var
        self.unbound_func_name = unbound_func_name

    def __call__(self, span, *args, **kwargs):
        return self.call_func(span, *args, **kwargs)

    def as_user_function(self, resource_handle, span=None):
        if span is None:
            span = _ir.Span()
        func_addr = "(MATXScriptBackendPackedCFunc)" + self.unbound_func_name + "__c_api"
        if resource_handle.same_as(self.session_pointer_var):
            # in class scope, symbol is a pointer
            sym_ref = _ir.call_extern(
                _ir.UserDataType(),
                "CAST_TO_USER_DATA_REF",
                span,
                self.symbol)
        else:
            # in other scope, symbol is a SharedView
            sym_ref = _ir.HLOCast(_ir.UserDataType(), self.symbol, span)
        value = _ir.call_extern(_ir.UserDataType(),
                                "MakeUserFunction",
                                span,
                                _ir.InitializerList([sym_ref], span),
                                _ir.StringImm(self.cls_name + "." + self.attr_name),
                                _ir.EnumAttr(func_addr),
                                resource_handle)
        return value


class ClassContext:

    def __init__(
            self,
            cls_name: str = '<unknown>',
            init_fn: Optional[FunctionContext] = None,
            methods: Optional[Dict[str, FunctionContext]] = None,
    ):
        self.raw: Optional[type] = None
        self.cls_id = 0
        self.cls_name = cls_name
        self.init_fn = init_fn
        self.methods = methods or {}

        self.attr_names: List[str] = []
        self.attr_types: Dict[str, AnnotatedType] = {}
        self.slot_names: List[str] = []
        self.annotation_slot_names: List[str] = []
        self.annotation_slot_ann: AnnotatedType = None
        self.is_abc: bool = False  # a class is called abc-class if it has a base of abc.ABC or abc-class
        self.abstract_methods: Set[str] = set()  # abstract methods of this class and its bases

        self._schema_updated = False
        self.class_instance_var = None  # c++ this pointer
        self.session_pointer_var = None  # pipeline session pointer
        self.base_ctx: Optional[ClassContext] = None

    @property
    def name(self):
        return self.cls_name

    def __str__(self):
        indent = lambda n: ' ' * 4 * n
        return '\n'.join([
            f'class {self.cls_name}:',
            indent(1) + str(self.init_fn),
            '\n'.join([
                (indent(2) + f'self.{attr_name}: {self.attr_types.get(attr_name)} = ...') for attr_name in
                self.attr_names
            ]),
            '\n'.join([
                (indent(1) + str(method_ctx)) for method_ctx in self.methods.values()
            ]),
        ])

    def init_ir_schema(self, base=None) -> _ir.adt.ClassType:
        header = _ir.GlobalTypeVar(self.name)
        schema = _ir.adt.ClassType(self.cls_id, header, base, [], [], [], [], [])
        return schema

    def update_ir_schema(self, schema) -> _ir.adt.ClassType:
        if self._schema_updated:
            raise Exception(
                'Trying to update class schema twice from the class context, which is illegal')
        self._schema_updated = True

        for name, ty in self.attr_types.items():
            schema.append_var(name, ty)
        arg_types = [ty for _, ty in self.init_fn.arg_types.items()]
        arg_types = [schema] + arg_types
        func_type = _ir.FuncType(arg_types, self.init_fn.return_type)
        schema.append_function(self.init_fn.name, self.init_fn.unbound_name, func_type)
        for name, fn_ctx in self.methods.items():
            arg_types = [ty for _, ty in fn_ctx.arg_types.items()]
            if fn_ctx.fn_type == FunctionType.INSTANCE:
                arg_types = [schema] + arg_types
            func_type = _ir.FuncType(arg_types, fn_ctx.return_type)
            schema.append_function(fn_ctx.name, fn_ctx.unbound_name, func_type)
        schema.rebuild_tag()
        return schema

    def ir_call_attr(self, span, symbol, attr):
        from ... import ir as _ir
        if self.base_ctx and "::" in attr and not attr.startswith(self.name + "::"):
            return self.base_ctx.ir_call_attr(span, symbol, attr)
        raw_attr = attr
        if "::" in attr and attr.startswith(self.name + "::"):
            raw_attr = attr[attr.find("::") + 2:]
        if raw_attr in self.attr_types:
            # self.var
            return _ir.op.object_get_attr(span, symbol, attr)
        elif raw_attr == '__init__' or raw_attr in self.methods:
            # self.func()
            err_msg_t = "could not convert the {}-th parameter type from '{}' to '{}'"
            if raw_attr == '__init__':
                func_ctx = self.init_fn
            else:
                func_ctx = self.methods[raw_attr]
            unbound_func_name = func_ctx.unbound_name
            func_ret_type = func_ctx.return_type
            bound_self = func_ctx.fn_type == FunctionType.INSTANCE

            def method_wrapper(span, *args, **kwargs):
                # assert hasattr(self.raw, raw_attr)
                sig = inspect.signature(getattr(self.raw, raw_attr))
                if bound_self:
                    binds = sig.bind(symbol, *args, **kwargs)
                    new_args = [binds.args[i] for i in range(1, len(binds.args))]
                else:
                    binds = sig.bind(*args, **kwargs)
                    new_args = binds.args
                if len(binds.kwargs) != 0:
                    binds.apply_defaults()
                smart_args = []
                for i in range(len(new_args)):
                    expect_ty = func_ctx.arg_types[func_ctx.arg_names[i]]
                    binds_arg_i = new_args[i]
                    if not isinstance(binds_arg_i, _ir.BaseExpr):
                        if binds_arg_i is None:
                            binds_arg_i = _ir.NoneExpr()
                        else:
                            assert isinstance(binds_arg_i, (bool, int, float, str, bytes))
                            binds_arg_i = _ir.const(binds_arg_i)
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
                if bound_self:
                    return _ir.Call(func_ret_type,
                                    _ir.op.ClassGetItem(symbol, attr, span),
                                    smart_args,
                                    span)
                else:
                    return _ir.call_extern(func_ret_type,
                                           unbound_func_name,
                                           span,
                                           *smart_args)

            return GetClassAttr(self.cls_name,
                                attr,
                                func_ctx,
                                symbol,
                                method_wrapper,
                                unbound_func_name,
                                self.session_pointer_var)
        else:
            if self.base_ctx:
                return self.base_ctx.ir_call_attr(span, symbol, attr)
            raise RuntimeError(
                '"{}" is not a member function or a member var of class "{}"'.format(
                    attr, self.name
                )
            )
