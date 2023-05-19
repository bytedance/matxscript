# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement:
# The structure of the expressions is inspired by Halide/TVM IR.
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
"""Function data types."""

from enum import IntEnum
from .. import _ffi
from .. import runtime
from ..runtime import Object
from .expr import PrimVar, Call
from . import _ffi_api
from .base import Stmt, Span
from ._converter import to_ir_object as _to_ir


class CallingConv(IntEnum):
    """Possible kinds of calling conventions."""

    DEFAULT = 0
    C_PACKED_FUNC = 1
    DEVICE_KERNEL_LAUNCH = 2


class FuncAttr(object):
    """Possible kinds of function attribute."""

    kCallingConv = "calling_conv"
    kGlobalSymbol = "global_symbol"
    kBoundSymbol = "bound_symbol"
    kExportSymbol = "export_symbol"
    kClassConstructor = "class_constructor"
    kClassNameBelongTo = "class_name_belong_to"
    kCaptureSessionHandle = "capture_session_handle"
    kKernelFunctionParameterBinding = "kernel_function_parameter_binding"


class BaseFunc(Stmt):
    """Base class of all functions."""

    @property
    def attrs(self):
        """Return the attrs member of the function."""
        return _ffi_api.BaseFunc_Attrs(self)

    def with_attr(self, attr_key_or_dict, attr_value=None):
        """Create a new copy of the function and update the attribute.

        Parameters
        ----------
        attr_key_or_dict : Union[str, dict]
            The attribute key to use or a dict containing multiple key value pairs.

        attr_value : Any
            The new attribute value.

        Returns
        -------
        func : Function
            A new copy of the function
        """
        # make sure we first copy so that we can safely do copy on write
        # for multiple updates.
        res = _ffi_api.BaseFuncCopy(self)

        if isinstance(attr_key_or_dict, dict):
            for key, val in attr_key_or_dict.items():
                res = _ffi_api.BaseFuncWithAttr(res._move(), _to_ir(key), _to_ir(val))
            return res

        return _ffi_api.BaseFuncWithAttr(
            res._move(), _to_ir(attr_key_or_dict), _to_ir(attr_value)
        )

    def get_func_type(self):
        return _ffi_api.BaseFunc_GetFuncType(self)


@_ffi.register_object("ir.PrimFunc")
class PrimFunc(BaseFunc):
    """A function declaration expression.

    Parameters
    ----------
    params: List[Union[ir.PrimVar, ir.Buffer]]
        List of input parameters to the function.

    default_params: List[ir.PrimExpr]
        List of the default input parameters to the function.

    body: ir.Stmt
        The body of the function.

    ret_type: ir.Type
        The return type annotation of the function.

    attrs: Optional[ir.Attrs]
        Attributes of the function, can be None
    """

    def __init__(self, params, default_params, body, ret_type=None, attrs=None):
        param_list = []
        for x in params:
            x = _to_ir(x) if not isinstance(x, Object) else x
            if isinstance(x, PrimVar):
                param_list.append(x)
            else:
                raise TypeError("params can only contain Var or Buffer")

        self.__init_handle_by_constructor__(
            _ffi_api.PrimFunc,
            _to_ir(param_list),
            _to_ir(default_params),
            _to_ir(body),
            _to_ir(ret_type),
            _to_ir(attrs)
        )

    def with_body(self, new_body):
        """Create a new PrimFunc with the same set signatures but a new body.

        Parameters
        ----------
        new_body : Stmt
            The new body.

        Returns
        -------
        new_func : PrimFunc
            The created new function.
        """
        return PrimFunc(self.params, _to_ir(new_body), self.ret_type, self.attrs)


@_ffi.register_object("ir.Function")
class Function(BaseFunc):
    """A function declaration expression.

    Parameters
    ----------
    params: List[ir.Var]
        List of input parameters to the function.

    default_params: List[ir.BaseExpr]
        List of the default input parameters to the function.

    body: ir.Stmt
        The body of the function.

    ret_type: Optional[ir.Type]
        The return type annotation of the function.

    type_params: Optional[List[ir.TypeParam]]
        The additional type parameters, this is only
        used in advanced usecase of template functions.
    """

    def __init__(self,
                 params,
                 default_params,
                 body,
                 ret_type=None,
                 type_params=None,
                 attrs=None,
                 span=Span()):
        if type_params is None:
            type_params = _to_ir([])

        self.__init_handle_by_constructor__(
            _ffi_api.Function,
            _to_ir(params),
            _to_ir(default_params),
            _to_ir(body),
            _to_ir(ret_type),
            _to_ir(type_params),
            _to_ir(attrs),
            _to_ir(span)
        )


@_ffi.register_object("ir.LambdaFunction")
class LambdaFunction(BaseFunc):
    """A lambda function declaration expression.

    Parameters
    ----------
    captures: List[ir.Var]
        List of input parameters to the function.

    params: List[ir.BaseExpr]
        List of the default input parameters to the function.

    body: ir.Stmt
        The body of the function.

    ret_type: Optional[ir.Type]
        The return type annotation of the function.
    """

    def __init__(self,
                 captures,
                 params,
                 body,
                 ret_type=None,
                 span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.LambdaFunction,
            _to_ir(captures),
            _to_ir(params),
            _to_ir(body),
            _to_ir(ret_type),
            _to_ir(span)
        )

    def __call__(self, span, *args):
        """Invoke the local function.

        Parameters
        ----------
        args: List[Expr]
            Arguments.
        """
        from .op_expr import Op
        return Call(self.ret_type, Op.get("ir.call_lambda"), _to_ir((self,) + args), span)
