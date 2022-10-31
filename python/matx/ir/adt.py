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
# pylint: disable=invalid-name
"""Algebraic data type definitions."""
from .. import _ffi

from .type import Type, ObjectType
from .expr import HLOExpr, Call, InitializerList, InitializerDict
from . import _ffi_api
from ._converter import to_ir_object as _to_ir


@_ffi.register_object("ir.Constructor")
class Constructor(HLOExpr):
    """ADT constructor.

    Parameters
    ----------
    name_hint : str
        Name of constructor (only a hint).

    inputs : Optional[List[matx.ir.Type]]
        Input types.

    belong_to : Optional[GlobalTypeVar]
        Denotes which ADT the constructor belongs to.
    """

    def __init__(self, name_hint, inputs=None, belong_to=None, ret_type=None):
        if ret_type is None:
            ret_type = ObjectType()
        self.__init_handle_by_constructor__(_ffi_api.Constructor,
                                            _to_ir(ret_type),
                                            _to_ir(name_hint),
                                            _to_ir(inputs),
                                            _to_ir(belong_to))

    def __call__(self, span, *args):
        """Call the constructor.

        Parameters
        ----------
        args: List[BaseExpr]
            The arguments to the constructor.

        Returns
        -------
        call: Expr
            A call to the constructor.
        """

        args_lower = []
        for arg in args:
            if isinstance(arg, list):
                # to be reviewed
                args_lower.append(InitializerList(arg))
            elif isinstance(arg, dict):
                # to be reviewed
                args_lower.append(InitializerDict(arg))
            else:
                args_lower.append(arg)
        return Call(self.checked_type, self, args_lower, span)


@_ffi.register_object("ir.ClassType")
class ClassType(Type):
    """User Custom Class in matx ir.

    Parameters
    ----------
    py_type_id : int
        The id of raw python type.

    header : GlobalTypeVar
        The name of the ADT.
        ADTs with the same constructors but different names are
        treated as different types.

    base : Optional[Type]
        base class type

    var_names : List[str]
        member variable names

    var_types : List[Type]
        member variable types

    func_names : List[str]
        member function names

    unbound_func_names : List[str]
        unbound member function names

    func_types : List[FuncType]
        member function types
    """

    def __init__(self,
                 py_type_id,
                 header,
                 base,
                 var_names,
                 var_types,
                 func_names,
                 unbound_func_names,
                 func_types):
        assert isinstance(py_type_id, int)
        self.__init_handle_by_constructor__(_ffi_api.ClassType,
                                            py_type_id,
                                            _to_ir(header),
                                            _to_ir(base),
                                            _to_ir(var_names),
                                            _to_ir(var_types),
                                            _to_ir(func_names),
                                            _to_ir(unbound_func_names),
                                            _to_ir(func_types))

    def __getitem__(self, item):
        """Get User Class var or function type

        Parameters
        ----------
        item : str
            ClassType attribute name

        Returns
        -------
        result: Type
            Var type or member function type

        """
        r = _ffi_api.ClassType_GetItem(self, _to_ir(item))
        assert r is not None
        return r

    def append_function(self, name, unbound_name, func_type):
        """Inplace append function

        Parameters
        ----------
        name : str
            function name

        unbound_name : str
            unbound function name

        func_type : FuncType
            function schema

        Returns
        -------
        """
        _ffi_api.ClassType_InplaceAppendFunc(
            self,
            _to_ir(name),
            _to_ir(unbound_name),
            _to_ir(func_type)
        )

    def append_var(self, name, var_type):
        """Inplace append var

        Parameters
        ----------
        name : str
            variable name

        var_type : Type
            variable type

        Returns
        -------
        """
        _ffi_api.ClassType_InplaceAppendVar(self, _to_ir(name), _to_ir(var_type))

    def rebuild_tag(self, mask=None):
        """Rebuild class uniq id

        Parameters
        ----------
        mask : optional[int]
            tag mask

        Returns
        -------

        """
        if isinstance(mask, int):
            _ffi_api.ClassType_RebuildTag(self, mask)
        else:
            _ffi_api.ClassType_RebuildTag(self)

    def clear_members(self):
        _ffi_api.ClassType_ClearMembers(self)


def get_implicit_class_session_var():
    return _ffi_api.GetImplicitClassSessionVar()
