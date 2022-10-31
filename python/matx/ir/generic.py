# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The generic ops is inspired by TVM.
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

from typing import Callable
from .base import BaseExpr, HLOExpr, PrimExpr, Span
from .type import Type, ListType, PrimType, ObjectType, StringType, UnicodeType
from . import type_relation as _type_rel
from . import _ffi_api
from .. import _ffi
from ..runtime.object import Object

###############################################################################
# Generic arith OP
###############################################################################


def _cast_to_prim_expr(value: BaseExpr, span: Span = Span()):
    if isinstance(value, PrimExpr):
        return value
    caster = _ffi.get_global_func('ir.HLOCastPrim')
    return caster(value.checked_type.dtype, value, span)


def _is_prim_type(value: BaseExpr):
    if isinstance(value, PrimExpr):
        return True
    return isinstance(value.checked_type, PrimType)


def left_shift(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """lhs left_shift rhs bits.

    Parameters
    ----------
    lhs : BaseExpr
        The left hand operand

    rhs : BaseExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    err_msg = "unsupported arg type(s) for <<: '{}' and '{}'".format(
        lhs.checked_type, rhs.checked_type)
    lhs, rhs = convert_type(_cast_to_prim_int,
                            err_msg,
                            span,
                            lhs,
                            rhs)
    return _ffi_api.left_shift(lhs, rhs, span)


def right_shift(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """a right_shift b bits.

    Parameters
    ----------
    lhs : BaseExpr
        The left hand operand

    rhs : BaseExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    err_msg = "unsupported arg type(s) for >>: '{}' and '{}'".format(
        lhs.checked_type, rhs.checked_type)
    lhs, rhs = convert_type(_cast_to_prim_int,
                            err_msg,
                            span,
                            lhs,
                            rhs)
    return _ffi_api.right_shift(lhs, rhs, span)


def bitwise_and(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Compute the bitwise_and of two expressions.

    Parameters
    ----------
    lhs : BaseExpr
        The left hand operand

    rhs : BaseExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    err_msg = "unsupported arg type(s) for &: '{}' and '{}'".format(
        lhs.checked_type, rhs.checked_type)
    lhs, rhs = convert_type(_cast_to_prim_int,
                            err_msg,
                            span,
                            lhs,
                            rhs)
    return _ffi_api.bitwise_and(lhs, rhs, span)


def bitwise_or(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Compute the bitwise_or of two expressions.

    Parameters
    ----------
    lhs : BaseExpr
        The left hand operand

    rhs : BaseExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    err_msg = "unsupported arg type(s) for |: '{}' and '{}'".format(
        lhs.checked_type, rhs.checked_type)
    lhs, rhs = convert_type(_cast_to_prim_int,
                            err_msg,
                            span,
                            lhs,
                            rhs)
    return _ffi_api.bitwise_or(lhs, rhs, span)


def bitwise_xor(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Compute the bitwise_xor of two expressions.

    Parameters
    ----------
    lhs : BaseExpr
        The left hand operand

    rhs : BaseExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    err_msg = "unsupported arg type(s) for ^: '{}' and '{}'".format(
        lhs.checked_type, rhs.checked_type)
    lhs, rhs = convert_type(_cast_to_prim_int,
                            err_msg,
                            span,
                            lhs,
                            rhs)
    return _ffi_api.bitwise_xor(lhs, rhs, span)


def bitwise_not(val: BaseExpr, span: Span = Span()):
    """Compute the bitwise_not of input expression.

    Parameters
    ----------
    val : BaseExpr
        The input expression

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    err_msg = "unsupported arg type(s) for ~: '{}'".format(val.checked_type)
    val = convert_type(_cast_to_prim_int,
                       err_msg,
                       span,
                       val)
    return _ffi_api.bitwise_not(val, span)


def add(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic add operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of add operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for +: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpAdd(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpAdd(lhs, rhs, span)


def subtract(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic subtract operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of subtract operaton.
    """
    err_msg = "unsupported arg type(s) for -: '{}' and '{}'".format(
        lhs.checked_type, rhs.checked_type)
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpSub(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpSub(lhs, rhs, span)


def multiply(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic multiply operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of multiply operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for *: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpMul(lhs, rhs, span)
    else:
        from . import op as _op
        if (_type_rel.is_type_of(rhs, (PrimType.IntType, PrimType.BoolType))
                and _type_rel.is_type_of(lhs, (StringType, UnicodeType, ListType))):
            return _op.explicit_repeat(span, _type_rel.remove_view(lhs.checked_type), lhs, rhs)
        if (_type_rel.is_type_of(lhs, (PrimType.IntType, PrimType.BoolType))
                and _type_rel.is_type_of(rhs, (StringType, UnicodeType, ListType))):
            return _op.explicit_repeat(span, _type_rel.remove_view(rhs.checked_type), rhs, lhs)
        return _ffi_api._HLO_OpMul(lhs, rhs, span)


def divide(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic divide operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of divide operaton.
    """
    err_msg = "unsupported arg type(s) for /: '{}' and '{}'".format(
        lhs.checked_type, rhs.checked_type)
    lhs, rhs = convert_type(_cast_to_prim_float,
                            err_msg,
                            span,
                            lhs,
                            rhs)
    return _ffi_api._OpDiv(lhs, rhs, span)


def handle_error(span: Span, err_msg: str, err_type: Exception = TypeError):
    err_context = 'File "{}", line {}, in {}'.format(
        span.file_name.decode(), span.lineno, span.func_name.decode())
    typed_err_msg = '{}: {}'.format(err_type.__name__, err_msg)
    err_info = err_context + "\n" + span.source_code.decode() + "\n" + typed_err_msg
    raise RuntimeError(err_info)


def convert_type(
        cast_func: Callable,
        err_msg: str,
        span: Span,
        *vals: BaseExpr):
    try:
        new_vals = []
        for val in vals:
            new_vals.append(cast_func(val))
    except:
        handle_error(span, err_msg)

    return tuple(new_vals) if len(new_vals) > 1 else new_vals[0]


def floordiv(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic floordiv operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of divide operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for //: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpFloorDiv(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpFloorDiv(lhs, rhs, span)


def floormod(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic floormod operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of mod operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for %: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpFloorMod(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpFloorMod(lhs, rhs, span)


def cast(src: BaseExpr, dtype: Object, span: Span = Span()):
    """Generic cast operator.

    Parameters
    ----------
    src : BaseExpr
        The source operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of divide operaton.
    """
    return _ffi_api._cast(dtype, src, span)


def _cast_to_prim_int(value: BaseExpr, span: Span = Span()):
    from .expr import NoneExpr
    if isinstance(value, NoneExpr):
        raise TypeError("expect 'int' but get 'NoneType'")
    if isinstance(value, PrimExpr):
        return value
    if not _type_rel.type_convertible(value.checked_type, PrimType("int64")):
        raise TypeError(f"expect 'int' but get '{value.py_type_name()}'")
    caster = _ffi.get_global_func('ir.HLOCastPrim')
    return caster("int64", value, span)


def _cast_to_prim_bool(value: BaseExpr, span: Span = Span()):
    from .expr import NoneExpr
    if isinstance(value, NoneExpr):
        raise TypeError("expect 'bool' but get 'NoneType'")
    if isinstance(value, PrimExpr):
        return value
    if not _type_rel.type_convertible(value.checked_type, PrimType("int64")):
        raise TypeError(f"expect 'bool' but get '{value.py_type_name()}'")
    caster = _ffi.get_global_func('ir.HLOCastPrim')
    return caster("bool", value, span)


def _cast_to_prim_float(value: BaseExpr, span: Span = Span()):
    from .expr import NoneExpr
    if isinstance(value, NoneExpr):
        raise TypeError("expect 'float' but get 'NoneType'")
    if isinstance(value, PrimExpr):
        return value
    if not _type_rel.type_convertible(value.checked_type, PrimType("float64")):
        raise TypeError(f"expect 'float' but get '{value.py_type_name()}'")
    caster = _ffi.get_global_func('ir.HLOCastPrim')
    return caster("float64", value, span)


def op_and(span: Span, lhs: BaseExpr, rhs: BaseExpr, *args: BaseExpr):
    """Generic equal operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.
    args : (BaseExpr,), optional
        The more operand

    Returns
    -------
    op : matx.Expr
        The result Expr of equal operaton.
    """
    def type_convert_helper(lhs, rhs):
        err_msg = "unsupported arg type(s) for or: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        return convert_type(_cast_to_prim_bool,
                            err_msg,
                            span,
                            lhs,
                            rhs)

    if all(_type_rel.is_type_of(x, PrimType.BoolType) for x in (lhs, rhs, *args)):
        lhs, rhs = type_convert_helper(lhs, rhs)
        result = _ffi_api._OpAnd(lhs, rhs, span)
        for operand in args:
            result, operand = type_convert_helper(result, operand)
            result = _ffi_api._OpAnd(result, operand, span)
        return result
    else:
        result = _ffi_api._HLO_OpAnd(lhs, rhs, span)
        for operand in args:
            result = _ffi_api._HLO_OpAnd(result, operand, span)
        return result


def op_or(span: Span, lhs: BaseExpr, rhs: BaseExpr, *args: BaseExpr):
    """Generic equal operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.
    args : (BaseExpr,)
        The more operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of equal operaton.
    """
    def type_convert_helper(lhs, rhs):
        err_msg = "unsupported arg type(s) for or: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        return convert_type(_cast_to_prim_bool,
                            err_msg,
                            span,
                            lhs,
                            rhs)

    if all(_type_rel.is_type_of(x, PrimType.BoolType) for x in (lhs, rhs, *args)):
        lhs, rhs = type_convert_helper(lhs, rhs)
        result = _ffi_api._OpOr(lhs, rhs, span)
        for operand in args:
            result, operand = type_convert_helper(result, operand)
            result = _ffi_api._OpOr(result, operand, span)
        return result
    else:
        result = _ffi_api._HLO_OpOr(lhs, rhs, span)
        for operand in args:
            result = _ffi_api._HLO_OpOr(result, operand, span)
        return result


def op_not(val: BaseExpr, span: Span = Span()):
    """Generic equal operator.

    Parameters
    ----------
    val : BaseExpr
        The left operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of equal operaton.
    """
    err_msg = "unsupported arg type(s) for not: '{}'".format(val.checked_type)
    val = convert_type(_cast_to_prim_bool,
                       err_msg,
                       span,
                       val)
    return _ffi_api._OpNot(val, span)


def op_is(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic is operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of equal operaton.
    """
    from .expr import NoneExpr

    if not isinstance(rhs, NoneExpr):
        handle_error(span, "`is` clause supports `None` only.")
    if _is_prim_type(lhs):
        err_msg = "unsupported arg type(s) for is: '{}'".format(lhs.checked_type)
        lhs = convert_type(_cast_to_prim_expr,
                           err_msg,
                           span,
                           lhs)
    return _ffi_api._HLO_OpEQ(lhs, rhs, span)


def equal(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic equal operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of equal operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for ==: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpEQ(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpEQ(lhs, rhs, span)


def notequal(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic notequal operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of notequal operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for !=: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpNE(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpNE(lhs, rhs, span)


def greater_than(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic Greater than operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of Greater than operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for >: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpGT(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpGT(lhs, rhs, span)


def greater_or_equal(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic GreaterEqual operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of GreaterEqual operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for >=: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpGE(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpGE(lhs, rhs, span)


def less_than(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic Greater than operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of Greater than operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for <: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpLT(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpLT(lhs, rhs, span)


def less_or_equal(lhs: BaseExpr, rhs: BaseExpr, span: Span = Span()):
    """Generic GreaterEqual operator.

    Parameters
    ----------
    lhs : BaseExpr
        The left operand.
    rhs : BaseExpr
        The right operand.

    Returns
    -------
    op : matx.Expr
        The result Expr of GreaterEqual operaton.
    """
    if _is_prim_type(lhs) and _is_prim_type(rhs):
        err_msg = "unsupported arg type(s) for <=: '{}' and '{}'".format(
            lhs.checked_type, rhs.checked_type)
        lhs, rhs = convert_type(_cast_to_prim_expr,
                                err_msg,
                                span,
                                lhs,
                                rhs)
        return _ffi_api._OpLE(lhs, rhs, span)
    else:
        return _ffi_api._HLO_OpLE(lhs, rhs, span)
