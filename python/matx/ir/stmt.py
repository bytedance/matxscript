# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the Stmt is inspired by incubator-tvm.
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
"""Statement AST Node in MATX.

Each statement node have subfields that can be visited from python side.

.. code-block:: python

    x = ir.Var("n", "int32")
    a = ir.Var("array", "handle")
    st = ir.stmt.Store(a, x + 1, 1)
    assert isinstance(st, ir.stmt.Store)
    assert(st.buffer_var == a)
"""
from typing import Union
from .. import _ffi
from .base import Span

from ..runtime import Object
from . import _ffi_api
from ._converter import to_ir_object as _to_ir
from .generic import _cast_to_prim_bool, convert_type


def _make_string_imm(value):
    from .expr import StringImm, UnicodeImm
    if isinstance(value, StringImm):
        return value
    if isinstance(value, UnicodeImm):
        return StringImm(value.value)
    if isinstance(value, str):
        value = value.encode()
    assert isinstance(value, (bytes, bytearray))
    return StringImm(value)


class Stmt(Object):
    """Base class of all the statements."""


@_ffi.register_object("ir.ExprStmt")
class ExprStmt(Stmt):
    """AssignStmt node.

    Parameters
    ----------
    value : BaseExpr
        The value to be computed.
    """

    def __init__(self, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.ExprStmt, value, span)


@_ffi.register_object("ir.HLOYield")
class HLOYield(Stmt):
    """HLOYield expression.

    Parameters
    ----------
    symbol : ir.BaseExpr
        The symbol in the HLOYield
    """

    def __init__(self, symbol, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOYield, symbol, span)

    def astype(self, _):
        raise TypeError("astype cannot be used on HLOYield")


@_ffi.register_object("ir.AllocaVarStmt")
class AllocaVarStmt(Stmt):
    """AllocaVarStmt node.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: ir.Type
        The type annotation on the variable.

    init_value: ir.BaseExpr, optional
        The type annotation on the variable.
    """

    def __init__(self, name_hint, type_annotation, init_value=None, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.AllocaVarStmt,
                                            _to_ir(name_hint),
                                            type_annotation,
                                            init_value,
                                            span)
        self.__var = _ffi_api._GetVarFromAllocaVarStmt(self)

    @property
    def var(self):
        return self.__var


@_ffi.register_object("ir.AssignStmt")
class AssignStmt(Stmt):
    """AssignStmt node.

    Parameters
    ----------
    lhs : BaseExpr
        The value to be returned.
    rhs : BaseExpr
        The value to be returned.
    """

    def __init__(self, lhs, rhs, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.AssignStmt, lhs, rhs, span)


@_ffi.register_object("ir.ReturnStmt")
class ReturnStmt(Stmt):
    """ReturnStmt node.

    Parameters
    ----------
    value : BaseExpr
        The value to be returned.
    """

    def __init__(self, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.ReturnStmt, _to_ir(value), span)


@_ffi.register_object("ir.LetStmt")
class LetStmt(Stmt):
    """LetStmt node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : PrimExpr
        The value in to be binded.

    body : Stmt
        The body statement.
    """

    def __init__(self, var, value, body, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.LetStmt, var, value, body, span)


@_ffi.register_object("ir.AssertStmt")
class AssertStmt(Stmt):
    """AssertStmt node.

    Parameters
    ----------
    condition : PrimExpr
        The assert condition.

    message : PrimExpr
        The error message.

    body : Stmt
        The body statement.
    """

    def __init__(self, condition, message, body, span=Span()):
        message = _make_string_imm(message)
        self.__init_handle_by_constructor__(_ffi_api.AssertStmt, condition, message, body, span)


@_ffi.register_object("ir.For")
class For(Stmt):
    """For node.

    Parameters
    ----------
    loop_var : Var
        The loop variable.

    min_val : PrimExpr
        The begining value.

    max_val : PrimExpr
        The endding value.

    for_type : int
        The for type.

    device_api : int
        The device api type.

    body : Stmt
        The body statement.
    """

    Serial = 0
    Parallel = 1
    Vectorized = 2
    Unrolled = 3

    def __init__(
            self,
            loop_var,
            min_val,
            max_val,
            step_val,
            for_type,
            device_api,
            body,
            span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.For, loop_var, min_val, max_val, step_val, for_type, body, span
        )


@_ffi.register_object("ir.AutoFor")
class AutoFor(Stmt):
    """AutoFor node.

    Parameters
    ----------
    loop_vars : List[BaseExpr]
        The loop variable.

    container : BaseExpr
        The container value.

    body : Stmt
        The body statement.
    """

    def __init__(self, loop_vars, container, body, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.AutoFor, _to_ir(loop_vars), _to_ir(container), _to_ir(body), span
        )


@_ffi.register_object("ir.While")
class While(Stmt):
    """For node.

    Parameters
    ----------
    cond: PrimExpr
        condition.

    body : Stmt
        The body statement.
    """

    def __init__(self, cond, body, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.While, cond, body, span
        )


@_ffi.register_object("ir.Break")
class Break(Stmt):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Break)


@_ffi.register_object("ir.Continue")
class Continue(Stmt):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Continue)


@_ffi.register_object("ir.AttrStmt")
class AttrStmt(Stmt):
    """AttrStmt node.

    Parameters
    ----------
    node : Node
        The node to annotate the attribute

    attr_key : str
        Attribute type key.

    value : PrimExpr
        The value of the attribute

    body : Stmt
        The body statement.
    """

    def __init__(self, node, attr_key, value, body, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.AttrStmt, node, _to_ir(attr_key), value, body, span)


@_ffi.register_object("ir.SeqStmt")
class SeqStmt(Stmt):
    """Sequence of statements.

    Parameters
    ----------
    seq : list[Stmt]
        The statements
    """

    def __init__(self, seq, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.SeqStmt, _to_ir(seq), span)

    def __getitem__(self, i):
        return self.seq[i]

    def __len__(self):
        return len(self.seq)


@_ffi.register_object("ir.IfThenElse")
class IfThenElse(Stmt):
    """IfThenElse node.

    Parameters
    ----------
    condition : PrimExpr
        The expression

    then_case : Stmt
        The statement to execute if condition is true.

    else_case : Stmt
        The statement to execute if condition is false.
    """

    def __init__(self, condition, then_case, else_case, span=Span()):
        err_msg = "unsupported arg type(s) for if: '{}'".format(condition.checked_type)
        condition = convert_type(_cast_to_prim_bool,
                                 err_msg,
                                 span,
                                 condition)
        self.__init_handle_by_constructor__(
            _ffi_api.IfThenElse, condition, then_case, else_case, span)


@_ffi.register_object("ir.ExceptionHandler")
class ExceptionHandler(Stmt):
    """ExceptionHandler node.

    Parameters
    ----------
    e : BaseExpr
        The expression

    body : Stmt
        The statement to execute if catch e.
    """

    def __init__(self, e, body, span=Span()):
        assert e is None, "specific exception is not supported now!!!"
        self.__init_handle_by_constructor__(
            _ffi_api.ExceptionHandler, e, body, span
        )


@_ffi.register_object("ir.TryExcept")
class TryExcept(Stmt):
    """TryExcept node.

    Parameters
    ----------
    body : Stmt
        The statement to execute in a try context.

    handlers: list[ExceptionHandler]
        The exception handler statements.
    """

    def __init__(self, body, handlers, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.TryExcept, body, _to_ir(handlers), span
        )


@_ffi.register_object("ir.Raise")
class Raise(Stmt):
    """Raise node.

    Parameters
    ----------
    exc : BaseExpr
        The expr to throw.
    """

    def __init__(self, exc, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.Raise, _to_ir(exc), span
        )


@_ffi.register_object("ir.Evaluate")
class Evaluate(Stmt):
    """Evaluate node.

    Parameters
    ----------
    value : PrimExpr
        The expression to be evalued.
    """

    def __init__(self, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.Evaluate, value, span)


# @_ffi.register_object("ir.Prefetch")
# class Prefetch(Stmt):
#     """Prefetch node.
#
#     Parameters
#     ----------
#     buffer : Buffer
#         The buffer to be prefetched.
#
#     bounds : list of Range
#         The bounds to be prefetched.
#     """
#
#     def __init__(self, buffer, bounds):
#         self.__init_handle_by_constructor__(_ffi_api.Prefetch, buffer, bounds)


def stmt_seq(*args):
    """Make sequence of statements

    Parameters
    ----------
    args : list of Expr or Var
        List of statements to be combined as sequence.

    Returns
    -------
    stmt : Stmt
        The combined statement.
    """
    ret = []
    for value in args:
        if not isinstance(value, Stmt):
            value = Evaluate(value)
        ret.append(value)
    if len(ret) == 1:
        return ret[0]
    return SeqStmt(ret)


def stmt_list(stmt):
    """Make list of stmt from blocks.

    Parameters
    ----------
    stmt : A block statement

    Returns
    -------
    stmt_list : list of Stmt
         The unpacked list of statements
    """
    if isinstance(stmt, SeqStmt):
        res = []
        for x in stmt:
            res += stmt_list(x)
        return res
    return [stmt]
