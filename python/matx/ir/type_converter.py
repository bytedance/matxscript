# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The functional ops is inspired by incubator-tvm.
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
from functools import lru_cache
import types
from matx._typed_ast import ast
from typing import Optional

from . import type as _ty

_MODULE_NAME_ = "matx"
NAME_NOT_FOUND = object()


class TypeConvertException(Exception):
    exception_msg = 'TypeConvertException'

    def __init__(self, node, msg=None, *args: object) -> None:
        self.node: ast.AST = node
        msg = msg if msg is not None else self.exception_msg
        super().__init__(msg, *args)


class TypeAnnotationParseException(TypeConvertException):
    exception_msg = 'This type annotation is not supported now'


class TypeNotFoundException(TypeConvertException):
    exception_msg = 'Type declaration is not found.'


class _AnnTypeConvert(ast.NodeVisitor):

    def __init__(self, mod_ctx=None):
        self.ty_map = {
            "bool": lambda: _ty.PrimType("bool"),
            "int": lambda: _ty.PrimType("int64"),
            "float": lambda: _ty.PrimType("float64"),
            "int32": lambda: _ty.PrimType("int32"),
            "int64": lambda: _ty.PrimType("int64"),
            "uint32": lambda: _ty.PrimType("uint32"),
            "uint64": lambda: _ty.PrimType("uint64"),
            "{}.Dict".format(_MODULE_NAME_): lambda: _ty.DictType(),
            "{}.List".format(_MODULE_NAME_): lambda: _ty.ListType(),
            "{}.Set".format(_MODULE_NAME_): lambda: _ty.SetType(),
            "{}.String".format(_MODULE_NAME_): lambda: _ty.StringType(),
            "{}.File".format(_MODULE_NAME_): lambda: _ty.FileType(),
            "{}.Trie".format(_MODULE_NAME_): lambda: _ty.TrieType(),
            "{}.NativeData".format(_MODULE_NAME_): lambda: _ty.UserDataType(),
            "{}.NativeOp".format(_MODULE_NAME_): lambda: _ty.UserDataType(),
            "{}.NativeObject".format(_MODULE_NAME_): lambda: _ty.UserDataType(),
            "{}.NDArray".format(_MODULE_NAME_): lambda: _ty.NDArrayType(),
            "{}.Regex".format(_MODULE_NAME_): lambda: _ty.RegexType(),
            "{}.OpaqueObject".format(_MODULE_NAME_): lambda: _ty.OpaqueObjectType(),
            "bytes": lambda: _ty.StringType(),
            "str": lambda: _ty.UnicodeType(),
            "AnyStr": lambda: _ty.ObjectType(),
            "object": lambda: _ty.ObjectType(),
            "Any": lambda: _ty.ObjectType(),
            "iter": lambda: _ty.ObjectType(),  # TODO fix me
            "List": lambda *args: _ty.ListType(False, *args),
            "Set": lambda *args: _ty.SetType(False, *args),
            "Dict": lambda *args: _ty.DictType(False, *args),
            "Tuple": lambda *args: _ty.TupleType(args),
            "FTList": lambda *args: _ty.ListType(True, *args),
            "FTSet": lambda *args: _ty.SetType(True, *args),
            "FTDict": lambda *args: _ty.DictType(True, *args),
            "Callable": lambda: _ty.UserDataType(),
            "None": lambda: _ty.ObjectType(),
            "list": lambda: _ty.ListType(),
            "dict": lambda: _ty.DictType(),
            "set": lambda: _ty.SetType(),
            "tuple": lambda args: _ty.TupleType(args),
            "bytes_view": lambda: _ty.StringType(is_view=True),
            "unicode_view": lambda: _ty.UnicodeType(is_view=True),
            "any_view": lambda: _ty.ObjectType(is_view=True),
        }
        assert mod_ctx is None or hasattr(mod_ctx, "globals")
        self.mod_ctx = mod_ctx

    def convert(self, node, allow_empty_tuple=False):
        ty = self.visit(node)
        if isinstance(ty, types.FunctionType):
            ty = ty()
        if isinstance(ty, _ty.FuncType):
            return _ty.UserDataType()
        if isinstance(ty, _ty.TupleType):
            if not allow_empty_tuple and len(ty.fields) == 0:
                raise TypeAnnotationParseException(
                    node,
                    'Incomplete type annotation for Tuple, please use Tuple[...]')
        return ty

    @lru_cache(50)
    def convert_str(self, node_str, allow_empty_tuple=False):
        node = ast.parse(node_str)
        return self.convert(node.body[0].value, allow_empty_tuple)

    def visit(self, node):
        """Override method in ast.NodeVisitor"""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        visit_res = visitor(node)
        return visit_res

    def visit_Name(self, node):
        if self.mod_ctx is not None:
            mod = self.mod_ctx.globals.get(node.id, NAME_NOT_FOUND)
            if mod is not NAME_NOT_FOUND and mod in self.ty_map:
                return self.ty_map[mod]
        if hasattr(node, 'my_user_data_flag'):
            # userdata only need name as adt first arg
            return node.id
        try:
            return self.ty_map[node.id]
        except KeyError:
            raise TypeNotFoundException(node)

    def visit_Constant(self, node):
        if node.value is None:
            return self.ty_map['None']
        else:
            raise TypeNotFoundException(node)

    def visit_NameConstant(self, node):
        if node.value is None:
            return self.ty_map['None']
        else:
            raise TypeNotFoundException(node)

    def visit_Attribute(self, node: ast.Attribute):
        # TODO: fixme
        def flatten_nested_attrs(n: ast.expr, attr: str):
            if isinstance(n, ast.Name):
                return n.id + "." + attr
            elif isinstance(n, ast.Attribute):
                return flatten_nested_attrs(n.value, n.attr + "." + attr)
            else:
                raise TypeNotFoundException(node)

        ann_ty = flatten_nested_attrs(node.value, node.attr)
        nested_ann_types = ann_ty.split('.')

        if self.mod_ctx is not None:
            node_attr = self.mod_ctx.globals.get(nested_ann_types[0], NAME_NOT_FOUND)
            if node_attr is not NAME_NOT_FOUND:
                for nat in nested_ann_types[1:]:
                    node_attr = getattr(node_attr, nat, NAME_NOT_FOUND)
                    if node_attr is NAME_NOT_FOUND:
                        break
            if node_attr is not NAME_NOT_FOUND and node_attr in self.ty_map:
                return self.ty_map[node_attr]

        if ann_ty == '{}.Tuple'.format(_MODULE_NAME_):
            raise TypeNotFoundException(
                node,
                'Please use Tuple[...] or typing.Tuple[...] as annotation for matx.Tuple')
        try:
            return self.ty_map[ann_ty]
        except KeyError:
            if ann_ty.startswith("matx."):
                # TODO: improve me
                return self.ty_map["Any"]
            raise TypeNotFoundException(node)

    def visit_Subscript(self, node):
        my_slot_names = None
        if hasattr(node, 'my_slot_names'):
            my_slot_names = node.my_slot_names

        symbol = self.visit(node.value)
        if isinstance(node.slice, ast.Index):
            # compatible with typed_ast with Python 3.7 or older
            slice_ty = self.convert(node.slice.value)
        else:
            slice_ty = self.convert(node.slice)

        if isinstance(slice_ty, tuple):
            def func_wrapper():
                if my_slot_names is None:
                    return symbol(*slice_ty)
                else:
                    return symbol(*slice_ty, my_slot_names)

            return func_wrapper
        else:
            def func_wrapper():
                if my_slot_names is None:
                    return symbol(slice_ty)
                else:
                    return symbol(slice_ty, my_slot_names)

            return func_wrapper

    def visit_Tuple(self, node):
        return tuple(self.convert(element) for element in node.elts)

    def generic_visit(self, node):
        raise TypeNotFoundException(node)
