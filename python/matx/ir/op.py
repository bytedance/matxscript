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
# pylint: disable=redefined-builtin, invalid-name
"""Operators used in IR expression."""
import inspect
import sys

from . import _ffi_api
from . import type as _type
from . import type_relation as _type_rel
from ._converter import const
from ._converter import convert
from .adt import ClassType, Constructor
from .base import BaseExpr, PrimExpr
from .expr import HLOCast
from .expr import HLOEnumerate
from .expr import HLOZip
from .expr import InitializerList
from .expr import IntImm, StringImm, NoneExpr
from .expr import PrimCall, Call, UnicodeImm, EnumAttr, ClassGetItem
from .generic import _cast_to_prim_expr, _cast_to_prim_float, convert_type, handle_error, _is_prim_type
from .op_expr import Op
from .type_checker import check_int_or_generic
from .type_converter import _AnnTypeConvert
from .type_relation import smart_adapt_to

OriginTypeConvert = _AnnTypeConvert()


def call_intrin(dtype, func_name, span, *args):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return PrimCall(dtype, func_name, convert(args), span)


def call_extern(ret_type, func_name, span, *args):
    """Build expression by calling a extern function.

    Parameters
    ----------
    ret_type: matx.ir.Type
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : BaseExpr
        The call expression.
    """
    return Call(ret_type, Op.get("ir.call_extern"), convert((StringImm(func_name),) + args), span)


def any(span, *args):
    """Create a new expression of the union of all conditions in the arguments

    Parameters
    ----------
    args : *args
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    from .generic import op_or
    if not args:
        err_msg = "Any must take at least 1 argument"
        handle_error(span, err_msg, SyntaxError)
    if len(args) == 1:
        return args[0]
    return op_or(span, *args)


def all(span, *args):
    """Create a new expression of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : *args
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        err_msg = "All must take at least 1 argument"
        handle_error(span, err_msg, SyntaxError)
    if len(args) == 1:
        return args[0]
    from .generic import op_and
    return op_and(span, *args)


def min_value(span, dtype):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    Returns
    -------
    value : matx.Expr
        The minimum value of dtype.
    """
    return _ffi_api.min_value(dtype, span)


def max_value(span, dtype):
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    Returns
    -------
    value : matx.Expr
        The maximum value of dtype.
    """
    return _ffi_api.max_value(dtype, span)


def exp(span, x):
    """Take exponetial of input x.
    http://www.cplusplus.com/reference/cmath/exp/

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.exp] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.exp", span, x)


def exp2(span, x):
    """Calculate 2**x

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.exp2] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.exp2", span, x)


def exp10(span, x):
    """Calculate 10**x

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.exp10] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.exp10", span, x)


def erf(span, x):
    """Take gauss error function of the input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x.dtype, "ir.erf", span, x)


def tanh(span, x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.tanh] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.tanh", span, x)


def sigmoid(span, x):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x.dtype, "ir.sigmoid", span, x)


def log(span, x):
    """Take log of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.log] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.log", span, x)


def log2(span, x):
    """Take log2 of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.log2] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.log2", span, x)


def log10(span, x):
    """Take log10 of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.log10] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.log10", span, x)


def log1p(span, x):
    """Take log(x + 1) with respect to input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x.dtype, "ir.log1p", span, x)


def tan(span, x):
    """Take tan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.tan] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.tan", span, x)


def cos(span, x):
    """Take cos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.cos] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.cos", span, x)


def cosh(span, x):
    """Take cosh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.cosh] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.cosh", span, x)


def acos(span, x):
    """Take acos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.acos] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.acos", span, x)


def acosh(span, x):
    """Take acos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.acosh] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.acosh", span, x)


def sin(span, x):
    """Take sin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.sin] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.sin", span, x)


def sinh(span, x):
    """Take sinh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.sinh] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.sinh", span, x)


def asin(span, x):
    """Take asin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.asin] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.asin", span, x)


def asinh(span, x):
    """Take asinh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.asinh] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.asinh", span, x)


def atan(span, x):
    """Take atan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.atan] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.atan", span, x)


def atanh(span, x):
    """Take atanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.atanh] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.atanh", span, x)


def atan2(span, x1, x2):
    """Take arctan2(x1, x2).

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.atan2] arg must be a number, not ({}, {})".format(
        x1.py_type_name(), x2.py_type_name()
    )
    x1, x2 = convert_type(_cast_to_prim_float,
                          err_msg,
                          span,
                          x1,
                          x2)
    return call_intrin(x1.dtype, "ir.atan2", span, x1, x2)


def sqrt(span, x):
    """Take square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.sqrt] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return call_intrin(x.dtype, "ir.sqrt", span, x)


def rsqrt(span, x):
    """Take reciprocal of square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x.dtype, "ir.rsqrt", span, x)


def floor(span, x):
    """Take floor of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.floor] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return _ffi_api.floor(x, span)


def ceil(span, x):
    """Take ceil of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[math.ceil] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return _ffi_api.ceil(x, span)


def trunc(span, x):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.trunc(x, span)


def abs(span, x):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[abs] arg must be a number, not {}".format(
        x.py_type_name()
    )
    if _is_prim_type(x):
        x = convert_type(_cast_to_prim_expr,
                         err_msg,
                         span,
                         x)
        return _ffi_api.abs(x, span)
    else:
        if not isinstance(x.checked_type, _type.ObjectType):
            handle_error(span, err_msg, TypeError)
        return _ffi_api._HLO_OpAbs(x, span)


def round(span, x):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.round(x, span)


def nearbyint(span, x):
    """Round elements of the array to the nearest integer.
    This intrinsic uses llvm.nearbyint instead of llvm.round
    which is faster but will results different from te.round.
    Notably nearbyint rounds according to the rounding mode,
    whereas te.round (llvm.round) ignores that.
    For differences between the two see:
    https://en.cppreference.com/w/cpp/numeric/math/round
    https://en.cppreference.com/w/cpp/numeric/math/nearbyint

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.nearbyint(x, span)


def nextafter(span, x1, x2):
    """Return the next floating-point value after x1 towards x2.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x1.dtype, "ir.nextafter", span, x1, x2)


def hypot(span, x1, x2):
    """Equivalent to sqrt(x1**2 + x2**2), element-wise.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x1.dtype, "ir.hypot", span, x1, x2)


def copysign(span, x1, x2):
    """Change the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x1.dtype, "ir.copysign", span, x1, x2)


def ldexp(span, x1, x2):
    """Returns x1 * (2 ** x2).

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x1.dtype, "ir.ldexp", span, x1, x2)


def isnan(span, x):
    """Check if input value is Nan.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[isnan] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return _ffi_api.isnan(x, span)


def isfinite(span, x):
    """Check if input value is finite.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[isfinite] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return _ffi_api.isfinite(x, span)


def isinf(span, x):
    """Check if input value is infinite.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    err_msg = "[isinf] arg must be a number, not {}".format(
        x.py_type_name()
    )
    x = convert_type(_cast_to_prim_float,
                     err_msg,
                     span,
                     x)
    return _ffi_api.isinf(x, span)


def power(span, x, y):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    Returns
    -------
    z : PrimExpr
        The result.
    """
    err_msg = "[power] arg must be a number, not {} and {}".format(
        x.py_type_name(), y.py_type_name()
    )
    x, y = convert_type(_cast_to_prim_float,
                        err_msg,
                        span,
                        x,
                        y)
    return call_intrin(x.dtype, "ir.pow", span, x, y)


def popcount(span, x):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin(x.dtype, "ir.popcount", span, x)


def q_multiply_shift(span, x, y, q, s):
    """Execute a multiplication between two Q-numbers x and y
    followed by a right shift s. The mathematical expression is:

       out = round(x*y*2^-s)

    More about Q-numbers here: https://en.wikipedia.org/wiki/Q_(number_format)
    The rounding rule is to the nearest value, rounding half up
    (i.e., round(x.1) = x and round (x.5) = x+1)

    Parameters
    ----------
    x : PrimExpr
        First Q-number
    y : PrimExpr
        Second Q-number
    q : PrimExpr
        Number of fractional bits in x and y. Needs to be > 0
    s : PrimExpr
        Integer shift

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin("int32", "ir.q_multiply_shift", span, x, y, q, s)


def fmod(span, x, y):
    """Return the remainder of x divided by y with the same sign as x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.
    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return call_intrin(x.dtype, "ir.fmod", span, x, y)


def if_then_else(span, cond, t, f):
    """Conditional selection expression.

    Parameters
    ----------
    cond : BaseExpr
        The condition

    t : BaseExpr
        The result expression if cond is true.

    f : BaseExpr
        The result expression if cond is false.

    Returns
    -------
    result : Node
        The result of conditional expression.

    Note
    ----
    Unlike Select, if_then_else will not execute
    the branch that does not satisfy the condition.
    You can use it to guard against out of bound access.
    Unlike Select, if_then_else cannot be vectorized
    if some lanes in the vector have different conditions.
    """
    if (isinstance(cond, PrimExpr)
            and isinstance(t, PrimExpr)
            and isinstance(f, PrimExpr)):
        return _ffi_api._OpIfThenElse(convert(cond), convert(t), convert(f), span)
    else:
        return _ffi_api._HLOOpIfThenElse(convert(cond), convert(t), convert(f), span)


def div(span, a, b):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b).
    """
    return _ffi_api._OpDiv(a, b, span)


def indexdiv(span, a, b):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexDiv(a, b, span)


def indexmod(span, a, b):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexMod(a, b, span)


def truncdiv(span, a, b):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncDiv(a, b, span)


def truncmod(span, a, b):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncMod(a, b, span)


def floordiv(span, a, b):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorDiv(a, b, span)


def floormod(span, a, b):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorMod(a, b, span)


###############################################################################
# HLO OP
###############################################################################


def hlo_call_intrin_template(ret_type, func_name, span, type_args, *args, **kwargs):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    ret_type : Type
        The type of the result.

    func_name: str
        The intrinsic function name.

    span : Span
        source code info

    type_args: Any
        template arguments

    args : *
        Positional arguments.

    Returns
    -------
    call : HloExpr
        The call expression.
    """

    op = Op.get(func_name)
    num_args = len(args)
    num_op_arguments = len(op.arguments)
    num_inputs_max = op.num_inputs_max
    num_inputs = op.num_inputs
    has_args = num_op_arguments > 0 and op.arguments[-1].type_info == b'*args'
    if has_args:
        num_op_arguments -= 1
        num_inputs -= 1

    num_inputs_max = num_args if num_inputs_max == -1 else num_inputs_max
    num_inputs = num_args if num_inputs == -1 else num_inputs
    if not (num_inputs_max >= num_args >= num_inputs):
        if num_inputs == num_inputs_max:
            expect_num = f"{num_inputs}"
        else:
            expect_num = f"({num_inputs}, {num_inputs_max})"
        raise SyntaxError(
            f"{func_name} Expect ({expect_num}) arguments but get num is {num_args}"
        )

    assert op.num_inputs_max == -1 or num_op_arguments == op.num_inputs_max, \
        "op.arguments != op.num_inputs_max. Please report error to matxscript developers."
    new_args = []
    i = 0
    l = num_op_arguments if num_op_arguments < num_args else num_args
    while i < l:
        ty = op.arguments[i].type_info
        if ty in (b"<template>"):
            new_args.append(args[i])
        else:
            raw_accept_ty_list = ty.decode().split("|")
            accept_ty_list = []
            for sub_ty in raw_accept_ty_list:
                ty_node = OriginTypeConvert.convert_str(sub_ty, True)
                accept_ty_list.append(ty_node)
            num_accept_ty = len(accept_ty_list)
            assert num_accept_ty > 0, "internal error"
            for idx, sub_ty in enumerate(accept_ty_list):
                if idx == num_accept_ty - 1:
                    if _type_rel.type_convertible(args[i].checked_type, sub_ty):
                        new_args.append(smart_adapt_to(args[i], sub_ty))
                    else:
                        accept_ty_list = [x.py_type_name() for x in accept_ty_list]
                        raise TypeError(
                            f"{func_name}() expect Union[{', '.join(accept_ty_list)}], "
                            f"but get {args[i].py_type_name()}"
                        )
                else:
                    if _type_rel.is_type_of(args[i], _type.ObjectType):
                        # always select the last type
                        continue
                    if _type_rel.type_convertible(args[i].checked_type, sub_ty):
                        new_args.append(smart_adapt_to(args[i], sub_ty))
                        break
        i += 1
    rested_args = args[i:]
    if len(kwargs) != 0:
        last_arg = make_kwargs_op(span, **kwargs)
        rested_args = [*rested_args, last_arg]
    if has_args:
        new_args.append(InitializerList(rested_args))
    else:
        new_args.extend(rested_args)
    args = new_args
    return Call(ret_type, op, convert(args), span, type_args=type_args)


def hlo_call_intrin(ret_type, func_name, span, *args, **kwargs):
    return hlo_call_intrin_template(ret_type, func_name, span, None, *args, **kwargs)


###############################################################################
# HLO Generic OP
###############################################################################

def _container_type_name(container_expr):
    assert isinstance(container_expr, BaseExpr)
    name_map = {
        _type.ListType: 'list',
        _type.DictType: 'dict',
        _type.SetType: 'set',
        _type.StringType: 'str',
        _type.UnicodeType: 'unicode',
        _type.FileType: 'file',
        _type.TupleType: 'tuple',
        _type.UserDataType: 'user_data',
        _type.TrieType: 'trie',
        _type.NDArrayType: 'ndarray',
        _type.RegexType: 'regex'
    }
    name = name_map.get(type(container_expr.checked_type), 'object')
    if container_expr.checked_type.is_full_typed():
        return "ft_" + name
    return name


def _builtin_func_name(container_expr, attr):
    assert isinstance(container_expr, BaseExpr)
    assert isinstance(attr, str)
    return "ir." + _container_type_name(container_expr) + "_" + attr


# generic userdata dispatch
def object_call_attr(span, userdata, attr, *args, **kwargs):
    if len(kwargs) != 0:
        last_arg = make_kwargs_op(span, **kwargs)
        return object_call_attr(span, userdata, attr, *args, last_arg)
    if _type_rel.is_type_of(userdata, _type.UserDataType):
        op_name = _builtin_func_name(userdata, "call_attr")
        attr_expr = StringImm(attr)
        return hlo_call_intrin(_type.ObjectType(), op_name, span, userdata, attr_expr, *args)
    else:
        # TODO: rename to call_attr
        op_name = 'ir.object___dispatch__'
        func_name = StringImm(attr)
        return hlo_call_intrin(_type.ObjectType(), op_name, span, userdata, func_name, *args)


def object_call(span, userdata, *args, **kwargs):
    if len(kwargs) != 0:
        last_arg = make_kwargs_op(span, **kwargs)
        return object_call(span, userdata, *args, last_arg)
    assert _type_rel.is_type_of(userdata, (_type.UserDataType, _type.ObjectType))
    op_name = _builtin_func_name(userdata, "call")
    return hlo_call_intrin(_type.ObjectType(), op_name, span, userdata, *args)


def object_get_attr(span, userdata, attr):
    if _type_rel.is_type_of(userdata, ClassType):
        # self.xx
        return ClassGetItem(userdata, attr, span)
    ret_type = _type.ObjectType()
    if _type_rel.is_type_of(userdata, _type.UserDataType):
        ret_type = _type.ObjectType(is_view=True)
    op_name = _builtin_func_name(userdata, "__getattr__")
    attr_expr = StringImm(attr)
    return hlo_call_intrin(ret_type, op_name, span, userdata, attr_expr)


def object_set_attr(span, userdata, attr, item_expr):
    op_name = _builtin_func_name(userdata, "__setattr__")
    attr_expr = StringImm(attr)
    return hlo_call_intrin(_type.VoidType(), op_name, span, userdata, attr_expr, item_expr)


def object_len(span, container_expr):
    func_name = _builtin_func_name(container_expr, "__len__")
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, container_expr)


def object_contains(span, container_expr, item_expr):
    func_name = _builtin_func_name(container_expr, "__contains__")
    return hlo_call_intrin(_type.PrimType("bool"), func_name, span, container_expr, item_expr)


def _is_all_type(args, _type):
    for arg in args:
        if not _type_rel.is_type_of(arg, _type):
            return False
    return True


def min(span, *args):
    # TODO: support kwargs
    if len(args) == 1:
        func_name = 'ir.math_iterable_min'
        return hlo_call_intrin(_type.ObjectType(), func_name, span, args[0])
    if _is_all_type(args, _type_rel.PrimType.IntType):
        func_name = 'ir.math_int_min'
        return hlo_call_intrin(_type.PrimType("int64"), func_name, span, *args)
    elif _is_all_type(args, _type_rel.PrimType.FloatType):
        func_name = 'ir.math_double_min'
        return hlo_call_intrin(_type.PrimType("float64"), func_name, span, *args)
    else:
        func_name = 'ir.math_min'
        return hlo_call_intrin(_type.ObjectType(), func_name, span, *args)


def max(span, *args):
    # TODO: kwargs
    if len(args) == 1:
        func_name = 'ir.math_iterable_max'
        return hlo_call_intrin(_type.ObjectType(), func_name, span, args[0])
    if _is_all_type(args, _type_rel.PrimType.IntType):
        func_name = 'ir.math_int_max'
        return hlo_call_intrin(_type.PrimType("int64"), func_name, span, *args)
    elif _is_all_type(args, _type_rel.PrimType.FloatType):
        func_name = 'ir.math_double_max'
        return hlo_call_intrin(_type.PrimType("float64"), func_name, span, *args)
    else:
        func_name = 'ir.math_max'
        return hlo_call_intrin(_type.ObjectType(), func_name, span, *args)


# subscript
def object_get_item(span, container_expr, item_expr):
    ret = _ffi_api.TryFusedNDArrayGetItem(container_expr, item_expr)
    if ret is not None:
        return ret
    err_msg = "[__getitem__] key must be {}"
    container_type = container_expr.checked_type
    call_ty = _type.ObjectType()
    cast_ty = None
    if _type_rel.is_type_of(container_expr, _type.ListType):
        check_int_or_generic(span, item_expr, err_msg.format('int'))
        if container_type.is_full_typed():
            call_ty = container_type.item_type
        else:
            cast_ty = container_type.item_type
    if _type_rel.is_type_of(container_expr, _type.TupleType):
        if isinstance(item_expr, IntImm):
            cast_ty = container_expr.checked_type[item_expr.value]
        else:
            check_int_or_generic(span, item_expr, err_msg.format('int'))
    if _type_rel.is_type_of(container_expr, _type.DictType):
        if container_type.is_full_typed():
            call_ty = container_type.value_type
        else:
            cast_ty = container_type.value_type
    if _type_rel.is_type_of(container_expr, _type.StringType):
        call_ty = _type.PrimType("int64")
        check_int_or_generic(span, item_expr, err_msg.format('int'))
    if _type_rel.is_type_of(container_expr, _type.UnicodeType):
        call_ty = _type.UnicodeType()
        check_int_or_generic(span, item_expr, err_msg.format('int'))
    func_name = _builtin_func_name(container_expr, "__getitem__")
    ret = hlo_call_intrin(call_ty, func_name, span, container_expr, item_expr)
    if cast_ty is not None:
        return HLOCast(cast_ty, ret, span)
    return ret


def object_set_item(span, container_expr, key_expr, val_expr):
    ret = _ffi_api.TryFusedNDArraySetItem(container_expr, key_expr, val_expr)
    if ret is not None:
        return ret
    func_name = _builtin_func_name(container_expr, "__setitem__")
    if _type_rel.is_type_of(container_expr, _type.ListType):
        check_int_or_generic(span, key_expr, "[__setitem__] key must be 'int' type")
        exp_val_ty = container_expr.checked_type.item_type
        if not _type_rel.type_convertible(val_expr.checked_type, exp_val_ty):
            raise TypeError(
                f"__setitem__ expect right value type is '{exp_val_ty.py_type_name()}', "
                f"but get '{val_expr.py_type_name()}'"
            )
        if container_expr.checked_type.is_full_typed():
            val_expr = _type_rel.smart_adapt_to(val_expr, exp_val_ty, span)
    if _type_rel.is_type_of(container_expr, _type.DictType):
        exp_key_ty = container_expr.checked_type.key_type
        exp_val_ty = container_expr.checked_type.value_type
        if not _type_rel.type_convertible(key_expr.checked_type, exp_key_ty):
            raise TypeError(
                f"__setitem__ expect key type is {exp_key_ty.py_type_name()}, "
                f"but get '{key_expr.py_type_name()}'"
            )
        if container_expr.checked_type.is_full_typed():
            key_expr = _type_rel.smart_adapt_to(key_expr, exp_key_ty, span)
        if not _type_rel.type_convertible(val_expr.checked_type, exp_val_ty):
            raise TypeError(
                f"__setitem__ expect right value type is '{exp_val_ty.py_type_name()}', "
                f"but get '{val_expr.py_type_name()}'"
            )
        if container_expr.checked_type.is_full_typed():
            val_expr = _type_rel.smart_adapt_to(val_expr, exp_val_ty, span)
    return hlo_call_intrin(_type.VoidType(), func_name, span, container_expr, key_expr, val_expr)


def object_get_slice(span, container_expr, start_expr, end_expr, step_expr):
    func_name = _builtin_func_name(container_expr, "__getslice__")
    ret_ty = container_expr.checked_type
    return hlo_call_intrin(ret_ty, func_name, span, container_expr,
                           start_expr, end_expr, step_expr)


def object_set_slice(span, container_expr, start_expr, end_expr, rcontainer_expr):
    func_name = _builtin_func_name(container_expr, "__setslice__")
    return hlo_call_intrin(_type.VoidType(), func_name, span, container_expr,
                           start_expr, end_expr, rcontainer_expr)


def object_append(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "append")
    if _type_rel.is_type_of(container_expr, _type.ListType):
        assert len(kwargs) == 0, "list.append() takes no keyword arguments"
        exp_item_ty = container_expr.checked_type.item_type
        if len(args) != 1:
            raise TypeError(f"append() takes exactly one argument ({len(args)} given)")
        if not _type_rel.type_convertible(args[0].checked_type, exp_item_ty):
            raise TypeError(
                f"append() expect {exp_item_ty.py_type_name()}, but get {args[0].py_type_name()}"
            )
        ret_type = _type.VoidType()
        if container_expr.checked_type.is_full_typed():
            args = (_type_rel.smart_adapt_to(args[0], exp_item_ty, span),)
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_extend(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "extend")
    if _type_rel.is_type_of(container_expr, _type.ListType):
        assert len(kwargs) == 0, "list.extend() takes no keyword arguments"
        exp_item_ty = container_expr.checked_type.item_type
        if len(args) != 1:
            raise TypeError(f"extend() takes exactly one argument ({len(args)} given)")
        if args[0].checked_type.is_iterable():
            value_type = _type_rel.infer_iterator_value_type(args[0].checked_type)
            if not _type_rel.type_convertible(value_type, exp_item_ty):
                raise TypeError(
                    f"extend() expect Iterable[{exp_item_ty.py_type_name()}], "
                    f"but get {args[0].py_type_name()}"
                )
        else:
            raise TypeError(f"'{args[0].py_type_name()}' object is not iterable")
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def explicit_repeat(span, ret_type, container_expr, int_expr):
    assert _type_rel.is_type_of(int_expr, (_type.PrimType.IntType, _type.PrimType.BoolType))
    if isinstance(container_expr, Call) and isinstance(container_expr.op, Constructor):
        # try fuse constructor and repeat
        constructor_op = container_expr.op
        constructor_args = container_expr.args
        if len(constructor_args) == 1 and isinstance(constructor_args[0], InitializerList):
            constructor_arg = constructor_args[0]
            init_args = constructor_arg.fields
            if isinstance(ret_type, _type.ListType):
                if len(init_args) == 1:
                    func_name = _builtin_func_name(container_expr, "fused_repeat_one")
                    return hlo_call_intrin(ret_type, func_name, span, init_args[0], int_expr)
                else:
                    func_name = _builtin_func_name(container_expr, "fused_repeat_many")
                    return hlo_call_intrin(ret_type, func_name, span, constructor_arg, int_expr)
                # if ret_type.is_full_typed:
    func_name = _builtin_func_name(container_expr, "repeat")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, int_expr)


def object_add(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "add")
    if _type_rel.is_type_of(container_expr, _type.SetType):
        assert len(kwargs) == 0, "set.add() takes no keyword arguments"
        exp_item_ty = container_expr.checked_type.item_type
        if len(args) != 1:
            raise TypeError(f"add() takes exactly one argument ({len(args)} given)")
        if not _type_rel.type_convertible(args[0].checked_type, exp_item_ty):
            raise TypeError(
                f"add() expect {exp_item_ty.py_type_name()}, but get {args[0].py_type_name()}"
            )
        ret_type = _type.VoidType()
        if container_expr.checked_type.is_full_typed():
            args = (_type_rel.smart_adapt_to(args[0], exp_item_ty, span),)
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_clear(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "clear")
    if _type_rel.is_type_of(container_expr, (_type.ListType, _type.DictType, _type.SetType)):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.clear() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_find(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "find")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        assert len(kwargs) == 0, "find() takes no keyword arguments"
        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_lower(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "lower")
    if _type_rel.is_type_of(container_expr, (_type.UnicodeType, _type.StringType)):
        assert len(kwargs) == 0, "lower() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_upper(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "upper")
    if _type_rel.is_type_of(container_expr, (_type.UnicodeType, _type.StringType)):
        assert len(kwargs) == 0, "lower() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_isdigit(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "isdigit")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        assert len(kwargs) == 0, "isdigit() takes no keyword arguments"
        ret_type = _type.PrimType("bool")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_isalpha(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "isalpha")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        assert len(kwargs) == 0, "isalpha() takes no keyword arguments"
        ret_type = _type.PrimType("bool")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_decode(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "decode")

    # def bytes_decode(encoding='utf-8', errors='strict'):
    def bytes_decode(encoding='utf-8', errors='strict'):
        assert not isinstance(encoding, BaseExpr), "decode() encoding argument is not supported now"
        assert not isinstance(errors, BaseExpr), "decode() errors argument is not supported now"
        ret_type = _type.UnicodeType()
        return hlo_call_intrin(ret_type, func_name, span, container_expr)

    if _type_rel.is_type_of(container_expr, _type.StringType):
        return bytes_decode(*args, **kwargs)
    else:
        return hlo_call_intrin(_type.ObjectType(), func_name, span, container_expr, *args, **kwargs)


def object_encode(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "encode")

    # def str_encode(encoding='utf-8', errors='strict'):
    def str_encode(encoding='utf-8', errors='strict'):
        assert not isinstance(encoding, BaseExpr), "encode() encoding argument is not supported now"
        assert not isinstance(errors, BaseExpr), "encode() errors argument is not supported now"
        ret_type = _type.StringType()
        return hlo_call_intrin(ret_type, func_name, span, container_expr)

    if _type_rel.is_type_of(container_expr, _type.UnicodeType):
        return str_encode(*args, **kwargs)
    else:
        return hlo_call_intrin(_type.ObjectType(), func_name, span, container_expr, *args, **kwargs)


def object_split(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "split")

    def regex_split(string):
        if _type_rel.is_type_of(string, _type.StringType):
            ret_type = _type.ListType(False, _type.StringType())
        elif _type_rel.is_type_of(string, _type.UnicodeType):
            ret_type = _type.ListType(False, _type.UnicodeType())
        else:
            ret_type = _type.ListType()
        return hlo_call_intrin(ret_type, func_name, span, container_expr, string)

    def str_split(sep=None, maxsplit=-1):
        if not isinstance(sep, BaseExpr):
            assert sep is None, "internal error"
            sep = NoneExpr()
        if not isinstance(maxsplit, BaseExpr):
            assert isinstance(maxsplit, int), "internal error"
            maxsplit = const(maxsplit, "int64")
        ret_type = _type.ListType(False, _type.UnicodeType())
        return hlo_call_intrin(ret_type, func_name, span, container_expr, sep, maxsplit)

    def bytes_split(sep=None, maxsplit=-1):
        if not isinstance(sep, BaseExpr):
            assert sep is None, "internal error"
            sep = NoneExpr()
        if not isinstance(maxsplit, BaseExpr):
            assert isinstance(maxsplit, int), "internal error"
            maxsplit = const(maxsplit, "int64")
        ret_type = _type.ListType(False, _type.StringType())
        return hlo_call_intrin(ret_type, func_name, span, container_expr, sep, maxsplit)

    if _type_rel.is_type_of(container_expr, _type.UnicodeType):
        return str_split(*args, **kwargs)
    elif _type_rel.is_type_of(container_expr, _type.StringType):
        return bytes_split(*args, **kwargs)
    elif _type_rel.is_type_of(container_expr, _type.RegexType):
        return regex_split(*args, **kwargs)
    else:
        return hlo_call_intrin(_type.ObjectType(), func_name, span, container_expr, *args, **kwargs)


def str_split_ft(span, container_expr, item_type, *args):
    func_name = _builtin_func_name(container_expr, "split_ft")
    ret_type = _type.ListType(is_full_typed=True, item_type=item_type)
    return Call(ret_type, Op.get(func_name), convert((container_expr, *args)),
                span=span, type_args=(item_type,))


def object_join(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "join")
    if _type_rel.is_type_of(container_expr, _type.StringType):
        assert len(kwargs) == 0, "bytes.join() takes no keyword arguments"
        ret_type = _type.StringType()
    if _type_rel.is_type_of(container_expr, _type.UnicodeType):
        assert len(kwargs) == 0, "str.join() takes no keyword arguments"
        ret_type = _type.UnicodeType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_startswith(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "startswith")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        assert len(kwargs) == 0, "startswith() takes no keyword arguments"
        ret_type = _type.PrimType("bool")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_endswith(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "endswith")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        assert len(kwargs) == 0, "endswith() takes no keyword arguments"
        ret_type = _type.PrimType("bool")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_rstrip(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "rstrip")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.rstrip() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_lstrip(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "lstrip")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.lstrip() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_strip(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "strip")
    if _type_rel.is_type_of(container_expr, (_type.StringType, _type.UnicodeType)):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.strip() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_count(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "count")
    ret_int_types = (_type.StringType, _type.UnicodeType, _type.ListType, _type.TupleType)
    if _type_rel.is_type_of(container_expr, ret_int_types):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.count() takes no keyword arguments"
        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_format(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "format")
    if _type_rel.is_type_of(container_expr, _type.UnicodeType):
        # TODO: fix unicode.format
        ret_type = _type.UnicodeType()
        args = InitializerList(args)
        return Call(ret_type,
                    Op.get(func_name),
                    convert((container_expr, args)))
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_readline(span, container_expr, *args, **kwargs):
    # TODO: fix file
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "readline")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_readlines(span, container_expr, *args, **kwargs):
    # TODO: fix file
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "readlines")
    if _type_rel.is_type_of(container_expr, _type.FileType):
        ret_type = _type.ListType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_read(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    cons_ty = container_expr.checked_type
    if isinstance(cons_ty, _type.FileType):
        assert len(kwargs) == 0, f"file.read() takes no keyword arguments"
        if cons_ty.binary_mode:
            ret_type = _type.StringType()
            func_name = _builtin_func_name(container_expr, "read_bytes")
        else:
            ret_type = _type.UnicodeType()
            func_name = _builtin_func_name(container_expr, "read_unicode")
    else:
        func_name = _builtin_func_name(container_expr, "read")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def builtins_open(span, path, mode=None, encoding=None):
    binary_mode = False
    if mode is None:
        mode = UnicodeImm('r')
    if not isinstance(mode, (UnicodeImm, StringImm)):
        raise TypeError('open(path, mode): mode must be a constant str')
    if mode.value == b'rb' and encoding is not None:
        raise TypeError('binary mode doesn\'t take an encoding argument')
    if encoding is None:
        encoding = UnicodeImm('utf8')
    if mode.value == b'rb':
        binary_mode = True
    ret_type = _type.FileType(binary_mode)
    return hlo_call_intrin(ret_type, 'ir.file_open', span, path, mode, encoding)


def object_close(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "close")
    if _type_rel.is_type_of(container_expr, _type.FileType):
        assert len(kwargs) == 0, "close() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_reserve(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "reserve")
    if _type_rel.is_type_of(container_expr, (_type.ListType, _type.SetType, _type.DictType)):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.reserve() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_capacity(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "capacity")
    if _type_rel.is_type_of(container_expr, (_type.ListType, _type.SetType, _type.DictType)):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.capacity() takes no keyword arguments"
        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_to_list(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "to_list")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.to_list() takes no keyword arguments"
        ret_type = _type.ListType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_tolist(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "tolist")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.tolist() takes no keyword arguments"
        ret_type = _type.ListType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_is_contiguous(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "is_contiguous")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.is_contiguous() takes no keyword arguments"
        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_contiguous(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "contiguous")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.contiguous() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_reshape(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "reshape")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.reshape() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_squeeze(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "squeeze")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.squeeze() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_unsqueeze(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "unsqueeze")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.unsqueeze() takes no keyword arguments"
        ret_type = container_expr.checked_type
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_shape(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "shape")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.shape() takes no keyword arguments"
        ret_type = _type.ListType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_dtype(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "dtype")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.dtype() takes no keyword arguments"
        ret_type = _type.UnicodeType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_dim(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "dim")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(args) == 0, f"{ty}.dim() takes no arguments ({len(args)} given)"
        assert len(kwargs) == 0, f"{ty}.dim() takes no keyword arguments"
        ret_type = _type.PrimType("int32")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_transpose(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "transpose")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.transpose() takes no keyword arguments"
        ret_type = _type.NDArrayType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_as_type(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "as_type")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.as_type() takes no keyword arguments"
        ret_type = _type.NDArrayType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_device(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "device")
    if _type_rel.is_type_of(container_expr, _type.NDArrayType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.device() takes no keyword arguments"
        ret_type = _type.UnicodeType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_bucket_count(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "bucket_count")
    if _type_rel.is_type_of(container_expr, (_type.SetType, _type.DictType)):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.bucket_count() takes no keyword arguments"
        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def unicodedata_normalize(span, form_expr, str_expr):
    func_name = 'ir.unicodedata_normalize'
    if isinstance(form_expr, UnicodeImm):
        form_s = form_expr.value.decode()
        form_map = {
            "NFC": 0,
            "NFKC": 1,
            "NFD": 2,
            "NFKD": 3,
        }
        assert form_s in form_map, "\"form\" should be one of ['NFC', 'NFKC', 'NFD', 'NFKD']"
        form = EnumAttr("UnicodeNormalForm::" + form_s)
    elif _type_rel.is_type_of(form_expr, _type_rel.UnicodeType):
        form = form_expr
    else:
        assert False, '"form" should be a Unicode.'
    return hlo_call_intrin(_type.UnicodeType(), func_name, span, form, str_expr)


def unicodedata_category(span, str_expr):
    func_name = 'ir.unicodedata_category'
    return hlo_call_intrin(_type.UnicodeType(), func_name, span, str_expr)


def object_update(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "update")
    if _type_rel.is_type_of(container_expr, _type.SetType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.update() takes no keyword arguments"
        args = InitializerList(args)
        ret_type = _type.VoidType()
        return Call(ret_type,
                    Op.get(func_name),
                    convert((container_expr, args)),
                    span=span)
    if _type_rel.is_type_of(container_expr, _type.TrieType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.update() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_prefix_search(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "prefix_search")
    if _type_rel.is_type_of(container_expr, _type.TrieType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.prefix_search() takes no keyword arguments"
        ret_type = _type.TupleType([_type.PrimType("int64"), _type.PrimType("int64")])
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_prefix_search_all(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "prefix_search_all")
    if _type_rel.is_type_of(container_expr, _type.TrieType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.prefix_search_all() takes no keyword arguments"
        ret_type = _type.ListType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_save(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "save")
    if _type_rel.is_type_of(container_expr, _type.TrieType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.save() takes no keyword arguments"
        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_load(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "load")
    if _type_rel.is_type_of(container_expr, _type.TrieType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.save() takes no keyword arguments"
        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_replace(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "replace")

    def str_replace(old, new, count=-1):
        if not isinstance(count, BaseExpr):
            assert isinstance(count, int), "internal error"
            count = const(count, "int64")
        ret_type = _type.UnicodeType()
        return hlo_call_intrin(ret_type, func_name, span, container_expr, old, new, count)

    def bytes_replace(old, new, count=-1):
        if not isinstance(count, BaseExpr):
            assert isinstance(count, int), "internal error"
            count = const(count, "int64")
        ret_type = _type.StringType()
        return hlo_call_intrin(ret_type, func_name, span, container_expr, old, new, count)

    def regex_replace(string, repl):
        ret_type = _type.ObjectType()
        if (_type_rel.is_type_of(string, _type.UnicodeType)
                or _type_rel.is_type_of(repl, _type.UnicodeType)):
            ret_type = _type.UnicodeType()
        elif (_type_rel.is_type_of(string, _type.StringType)
              or _type_rel.is_type_of(repl, _type.StringType)):
            ret_type = _type.StringType()
        return hlo_call_intrin(ret_type, func_name, span, container_expr, string, repl)

    if _type_rel.is_type_of(container_expr, _type.UnicodeType):
        return str_replace(*args, **kwargs)
    elif _type_rel.is_type_of(container_expr, _type.StringType):
        return bytes_replace(*args, **kwargs)
    elif _type_rel.is_type_of(container_expr, _type.RegexType):
        return regex_replace(*args, **kwargs)
    else:
        # Pack kwargs
        return hlo_call_intrin(_type.ObjectType(), func_name, span, container_expr, *args, **kwargs)


def object_match(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "match")

    def regex_match(string, offset=0):
        if not isinstance(offset, BaseExpr):
            assert isinstance(offset, int), "internal error"
            offset = const(offset, "int64")
        ret_type = _type.TupleType([_type.ListType(), _type.DictType()])
        return hlo_call_intrin(ret_type, func_name, span, container_expr, string, offset)

    if _type_rel.is_type_of(container_expr, _type.RegexType):
        return regex_match(*args, **kwargs)
    else:
        # Pack kwargs
        return hlo_call_intrin(_type.ObjectType(), func_name, span, container_expr, *args, **kwargs)


def object_keys(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "keys")
    if _type_rel.is_type_of(container_expr, _type.DictType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.keys() takes no keyword arguments"
        ret_type = _type.IteratorType(_type.ObjectType())
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_values(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "values")
    if _type_rel.is_type_of(container_expr, _type.DictType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.values() takes no keyword arguments"
        ret_type = _type.IteratorType(_type.ObjectType())
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_items(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "items")
    if _type_rel.is_type_of(container_expr, _type.DictType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.items() takes no keyword arguments"
        ret_type = _type.IteratorType(_type.ObjectType())
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_get(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "get")
    if _type_rel.is_type_of(container_expr, _type.DictType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.get() takes no keyword arguments"
        # TODO: fix type infer dict.get
    return hlo_call_intrin(_type.ObjectType(), func_name, span, container_expr, *args, **kwargs)


def json_loads(span, json_str):
    func_name = 'ir.json_loads'
    return hlo_call_intrin(_type.ObjectType(), func_name, span, json_str)


def json_load(span, fp):
    func_name = 'ir.json_load'
    return hlo_call_intrin(_type.ObjectType(), func_name, span, fp)


def json_dumps(span, obj, indent=None, ensure_ascii=True):
    func_name = 'ir.json_dumps'
    if indent is None:
        indent = -1
    return hlo_call_intrin(_type.UnicodeType(), func_name, span, obj, indent, ensure_ascii)


def object_pop(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "pop")
    container_type = container_expr.checked_type
    call_ty = _type.ObjectType()
    cast_ty = None
    num_args = len(args)
    if _type_rel.is_type_of(container_expr, _type.DictType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.pop() takes no keyword arguments"
        if num_args == 0:
            raise TypeError("pop expected at least 1 argument, got 0")
        elif num_args == 1:
            if container_type.is_full_typed():
                call_ty = container_type.value_type
            else:
                cast_ty = container_type.value_type
        elif num_args == 2:
            value_ty = container_type.value_type
            default_ty = args[1].checked_type
            if container_type.is_full_typed():
                call_ty = _type_rel.lift_type(value_ty, default_ty)
                if isinstance(call_ty, _type.ObjectType):
                    args = [args[0], HLOCast(_type.ObjectType(), args[1], span)]
            else:
                cast_ty = _type_rel.lift_type(value_ty, default_ty)
        else:
            raise TypeError("pop expected at most 2 arguments, got {}".format(num_args))
        if not container_type.is_full_typed():
            args = (InitializerList(args),)
        ret = Call(call_ty,
                   Op.get(func_name),
                   convert((container_expr, *args)),
                   span=span)
        if cast_ty is not None:
            return HLOCast(cast_ty, ret, span)
        return ret
    elif _type_rel.is_type_of(container_expr, _type.ListType):
        ty = container_type.py_type_name()
        assert len(kwargs) == 0, f"{ty}.pop() takes no keyword arguments"
        if num_args > 1:
            raise TypeError("pop expected at most 1 argument, got 2")
        if num_args == 1:
            check_int_or_generic(span, args[0])
        if container_type.is_full_typed():
            call_ty = container_type.item_type
        else:
            cast_ty = container_type.item_type
    ret = hlo_call_intrin(call_ty, func_name, span, container_expr, *args, **kwargs)
    if cast_ty is not None:
        return HLOCast(cast_ty, ret, span)
    return ret


def object_insert(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "insert")
    if _type_rel.is_type_of(container_expr, _type.ListType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.insert() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_index(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "index")
    if _type_rel.is_type_of(container_expr, _type.ListType):
        ty = container_expr.py_type_name()
        assert len(args) <= 3, f"{ty}.index() takes ony 3 arguments"
        assert len(kwargs) == 0, f"{ty}.index() takes no keyword arguments"

        ret_type = _type.PrimType("int64")
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_remove(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "remove")
    if _type_rel.is_type_of(container_expr, _type.ListType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.remove() takes no keyword arguments"
        ret_type = _type.VoidType()
    if _type_rel.is_type_of(container_expr, _type.SetType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.remove() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_reverse(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "reverse")
    if _type_rel.is_type_of(container_expr, _type.ListType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.reverse() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_sort(span, container_expr, *args, **kwargs):
    func_name = _builtin_func_name(container_expr, "sort")

    def _list_sort(key=None, reverse=False):
        ty = container_expr.py_type_name()
        assert len(args) == 0, f"{ty}.sort() takes no positional arguments"
        _ret_type = _type.VoidType()
        if not isinstance(reverse, BaseExpr):
            assert isinstance(reverse, bool), "internal error"
            reverse = const(reverse, "bool")
        if key is None:
            func_name_nk = _builtin_func_name(container_expr, "sort_no_key")
            return hlo_call_intrin(_ret_type, func_name_nk, span, container_expr, reverse)
        else:
            return hlo_call_intrin(_ret_type, func_name, span, container_expr, key, reverse)

    if _type_rel.is_type_of(container_expr, _type.ListType):
        return _list_sort(**kwargs)
    else:
        return hlo_call_intrin(_type.ObjectType(), func_name, span, container_expr, *args, **kwargs)


def object_difference(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "difference")
    if _type_rel.is_type_of(container_expr, _type.SetType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.difference() takes no keyword arguments"
        args = InitializerList(args)
        ret_type = _type.SetType()
        return Call(ret_type,
                    Op.get(func_name),
                    convert((container_expr, args)),
                    span=span)
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_union(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "union")
    if _type_rel.is_type_of(container_expr, _type.SetType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.union() takes no keyword arguments"
        args = InitializerList(args)
        ret_type = _type.SetType()
        return Call(ret_type,
                    Op.get(func_name),
                    convert((container_expr, args)),
                    span=span)
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_difference_update(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "difference_update")
    if _type_rel.is_type_of(container_expr, _type.SetType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.difference_update() takes no keyword arguments"
        args = InitializerList(args)
        ret_type = _type.VoidType()
        return Call(ret_type,
                    Op.get(func_name),
                    convert((container_expr, args)),
                    span=span)
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def object_discard(span, container_expr, *args, **kwargs):
    ret_type = _type.ObjectType()
    func_name = _builtin_func_name(container_expr, "discard")
    if _type_rel.is_type_of(container_expr, _type.SetType):
        ty = container_expr.py_type_name()
        assert len(kwargs) == 0, f"{ty}.discard() takes no keyword arguments"
        ret_type = _type.VoidType()
    return hlo_call_intrin(ret_type, func_name, span, container_expr, *args, **kwargs)


def time_time(span):
    func_name = 'ir.time_time'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span)


def base64_b64encode(span, s, altchars=None):
    func_name = 'ir.base64_b64encode'
    if altchars is None:
        altchars = NoneExpr()
    else:
        raise NotImplementedError("`altchars` is not supported now")
    return hlo_call_intrin(_type.StringType(), func_name, span, s, altchars)


def base64_b64decode(span, s, altchars=None, validate=False):
    func_name = 'ir.base64_b64decode'
    if altchars is None:
        altchars = NoneExpr()
    else:
        raise NotImplementedError("`altchars` is not supported now")
    if validate:
        raise NotImplementedError("`validate` is not supported now")
    validate = const(validate, "int64")
    return hlo_call_intrin(_type.StringType(), func_name, span, s, altchars, validate)


def os_getenv(span, key, default=None):
    func_name = 'ir.os_getenv'
    if not isinstance(default, BaseExpr):
        assert default is None, "internal error"
        default = NoneExpr()
    key = smart_adapt_to(key, _type.ObjectType(True), span)
    default = smart_adapt_to(default, _type.ObjectType(True), span)
    return hlo_call_intrin(_type.ObjectType(), func_name, span, key, default)


def builtins_sorted(span, iterable, key=None, reverse=False):
    func_name = 'ir.builtins_sorted'
    if key is None:
        key = NoneExpr()
    iterable_type = iterable.checked_type
    if _type_rel.is_type_of(iterable, _type.ListType):
        if iterable_type.is_full_typed():
            ret = hlo_call_intrin(
                _type.ListType(
                    is_full_typed=True,
                    item_type=iterable_type.item_type),
                func_name,
                span,
                iterable,
                key,
                reverse)
            return ret
        else:
            ret = hlo_call_intrin(_type.ListType(), func_name, span, iterable, key, reverse)
            return HLOCast(_type.ListType(item_type=iterable_type.item_type), ret, span)
    elif _type_rel.is_type_of(iterable, _type.TupleType):
        item_type = _type_rel.infer_iterator_value_type(iterable_type)
        return hlo_call_intrin(
            _type.ListType(
                item_type=item_type),
            func_name,
            span,
            iterable,
            key,
            reverse)
    else:
        ret = hlo_call_intrin(_type.ObjectType(), func_name, span, iterable, key, reverse)
        return HLOCast(_type.ListType(), ret, span)


# random
def random_random(span):
    func_name = 'ir.random_random'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span)


def random_seed(span, *args):
    func_name = 'ir.random_seed'
    return hlo_call_intrin(_type.VoidType(), func_name, span, *args)


def random_getstate(span):
    func_name = 'ir.random_getstate'
    # TODO: fix type_infer
    return hlo_call_intrin(_type.ObjectType(), func_name, span)


def random_setstate(span, state):
    func_name = 'ir.random_setstate'
    return hlo_call_intrin(_type.VoidType(), func_name, span, state)


def random_getrandbits(span, k):
    func_name = 'ir.random_getrandbits'
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, k)


def random_uniform(span, a, b):
    func_name = 'ir.random_uniform'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, a, b)


def random_triangular(span, *args):
    func_name = 'ir.random_triangular'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, *args)


def random_randint(span, a, b):
    func_name = 'ir.random_randint'
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, a, b)


def random_normalvariate(span, mu, sigma):
    func_name = 'ir.random_normalvariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, mu, sigma)


def random_lognormvariate(span, mu, sigma):
    func_name = 'ir.random_lognormvariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, mu, sigma)


def random_expovariate(span, lambd):
    func_name = 'ir.random_expovariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, lambd)


def random_vonmisesvariate(span, mu, kappa):
    func_name = 'ir.random_vonmisesvariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, mu, kappa)


def random_gammavariate(span, alpha, beta):
    func_name = 'ir.random_gammavariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, alpha, beta)


def random_gauss(span, mu, sigma):
    func_name = 'ir.random_gauss'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, mu, sigma)


def random_betavariate(span, alpha, beta):
    func_name = 'ir.random_betavariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, alpha, beta)


def random_paretovariate(span, alpha):
    func_name = 'ir.random_paretovariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, alpha)


def random_weibullvariate(span, alpha, beta):
    func_name = 'ir.random_weibullvariate'
    return hlo_call_intrin(_type.PrimType("float64"), func_name, span, alpha, beta)


# matx function
def matx_make_native_object(span, native_cls_name, *args):
    if isinstance(native_cls_name, str):
        native_cls_name = native_cls_name.encode()
    if isinstance(native_cls_name, (bytes, bytearray)):
        native_cls_name = StringImm(native_cls_name)
    if isinstance(native_cls_name, UnicodeImm):
        native_cls_name = StringImm(native_cls_name.value)
    assert isinstance(native_cls_name, StringImm)
    return call_extern(_type.UserDataType(), b"make_native_userdata", span, native_cls_name, *args)


def matx_make_native_op(span, op_cls, *args, **kwargs):
    op_name = op_cls.__name__
    init_info = inspect.getfullargspec(op_cls.__init__)
    init_arg_num = len(init_info.args)
    init_arg_names = set(init_info.args[1:])
    err_msg = "Parameters of %s does not match" % op_name
    args_dict = {}
    for i, arg in enumerate(args):
        if i + 1 >= init_arg_num:
            raise TypeError(err_msg)
        args_dict[init_info.args[i + 1]] = arg
    for k, arg in kwargs.items():
        if k not in init_arg_names:
            raise TypeError(err_msg)
        if k in args_dict:
            raise TypeError(err_msg)
        args_dict[k] = arg

    args = []
    args.append(StringImm(op_name))
    for k, v in args_dict.items():
        args.append(StringImm(k))
        args.append(v)
    return call_extern(_type.ObjectType(), b"make_native_op", span, *args)


def matx_make_native_function(span, native_func_name, *args):
    if isinstance(native_func_name, str):
        native_func_name = native_func_name.encode()
    if isinstance(native_func_name, (bytes, bytearray)):
        native_func_name = StringImm(native_func_name)
    if isinstance(native_func_name, UnicodeImm):
        native_func_name = StringImm(native_func_name.value)
    assert isinstance(native_func_name, StringImm)
    return call_extern(_type.UserDataType(), b"make_native_function", span, native_func_name)


def matx_call_native_function(span, native_func_name, *args):
    if isinstance(native_func_name, str):
        native_func_name = native_func_name.encode()
    if isinstance(native_func_name, (bytes, bytearray)):
        native_func_name = StringImm(native_func_name)
    if isinstance(native_func_name, UnicodeImm):
        native_func_name = StringImm(native_func_name.value)
    assert isinstance(native_func_name, StringImm)
    return call_extern(_type.ObjectType(), b"call_native_function", span, native_func_name, *args)


def matx_pmap(span, func, data, sess):
    input_ty = data.checked_type
    if isinstance(input_ty, _type.TupleType):
        num_inputs = len(input_ty.fields)
        result_type = _type.TupleType([_type.ObjectType()] * num_inputs)
    elif isinstance(input_ty, _type.ListType):
        result_type = _type.ListType()
    else:
        if not isinstance(input_ty, _type.ObjectType):
            raise TypeError(
                f"expect the second argument is list or tuple, but get '{input_ty.py_type_name()}'"
            )
        result_type = _type.ObjectType()
    func_ty = func.checked_type
    if not isinstance(func_ty, _type.UserDataType):
        func = smart_adapt_to(func, _type.UserDataType(), span)
    return call_extern(result_type, b"ParallelMap", span, func, data, sess)


def matx_pstarmap(span, func, data, sess):
    input_ty = data.checked_type
    if isinstance(input_ty, _type.TupleType):
        num_inputs = len(input_ty.fields)
        result_type = _type.TupleType([_type.ObjectType()] * num_inputs)
    elif isinstance(input_ty, _type.ListType):
        result_type = _type.ListType()
    else:
        if not isinstance(input_ty, _type.ObjectType):
            raise TypeError(
                f"expect the second argument is list or tuple, but get '{input_ty.py_type_name()}'"
            )
        result_type = _type.ObjectType()
    func_ty = func.checked_type
    if not isinstance(func_ty, _type.UserDataType):
        func = smart_adapt_to(func, _type.UserDataType(), span)
    return call_extern(result_type, b"ParallelStarMap", span, func, data, sess)


def matx_apply_async(span, func, *args):
    func_ty = func.checked_type
    if not isinstance(func_ty, _type.UserDataType):
        func = smart_adapt_to(func, _type.UserDataType(), span)
    result_type = _type.ObjectType()
    # TODO: fix type infer
    sess = args[-1]
    func_args = InitializerList(args[:-1])
    return call_extern(result_type, b"ApplyAsync", span, func, func_args, sess)


def builtins_unpack(span, index, lhs_ty, iterable):
    func_name = 'ir.builtins_unpack'
    return Call(_type.ObjectType(),
                Op.get(func_name),
                convert((iterable,)),
                type_args=(index, lhs_ty),
                span=span)


def builtins_enumerate(span, iterable, start=0):
    return HLOEnumerate(iterable, start, span)


def builtins_zip(span, *iterables):
    return HLOZip(iterables, span)


def builtins_print(span, *args, sep=' ', end='\n', file=None):
    func_name = 'ir.builtins_print'
    if file is None:
        file = EnumAttr("stdout")
    elif file == sys.stderr:
        file = EnumAttr("stderr")
    elif file == sys.stdout:
        file = EnumAttr("stdout")
    else:
        assert False, "print: 'file' arg supports sys.stderr and sys.stdout only"
    if isinstance(sep, (UnicodeImm, StringImm)):
        sep = sep.value
    assert isinstance(sep, (str, bytes, bytearray)), "print: 'sep' arg support str type only"
    if isinstance(end, (UnicodeImm, StringImm)):
        end = end.value
    assert isinstance(end, (str, bytes, bytearray)), "print: 'end' arg support str type only"
    sep = StringImm(sep)
    end = StringImm(end)
    # args = InitializerList(args)
    return Call(_type.ObjectType(),
                Op.get(func_name),
                convert((sep, end, file, *args)),
                span=span)


def builtins_ord(span, container_expr):
    func_name = 'ir.builtins_ord'
    return hlo_call_intrin(_type.PrimType("int64"), func_name, span, container_expr)


def builtins_chr(span, container_expr):
    func_name = 'ir.builtins_chr'
    return hlo_call_intrin(_type.UnicodeType(), func_name, span, container_expr)


def builtins_isinstance_1(span, container_expr, type_expr):
    func_name = 'ir.builtins_isinstance'
    cons_ty = container_expr.checked_type
    target_ty = type_expr.checked_type
    if not isinstance(cons_ty, _type.ObjectType):
        if isinstance(cons_ty, _type.PrimType):
            result = isinstance(target_ty, _type.PrimType) and cons_ty.dtype == target_ty.dtype
        else:
            result = isinstance(cons_ty, type(target_ty))
        return const(result, "bool")
    if isinstance(type_expr, Constructor):
        checked_type = target_ty
    elif inspect.isfunction(type_expr) and getattr(type_expr, 'is_constructor', False):
        checked_type = target_ty
    else:
        raise TypeError(f'isinstance: unsupported checked type')
    if isinstance(checked_type, _type.PrimType) and checked_type.dtype == 'bool':
        raise TypeError("isinstance: 'bool' is unsupported checked type, please use 'int' instead")
    if isinstance(checked_type, _type.TupleType):
        # TODO: clean me
        checked_type = _type.TupleType([_type.ObjectType()])
    return hlo_call_intrin_template(
        _type.PrimType("bool"),
        func_name,
        span,
        [checked_type],
        container_expr
    )


def builtins_isinstance(span, container_expr, type_expr):
    if isinstance(type_expr, Call) and isinstance(type_expr.op, Constructor):
        if len(type_expr.args) != 1 or not isinstance(type_expr.args[0], InitializerList):
            raise TypeError('isinstance: unsupported checked type')
        type_expr = type_expr.args[0].fields
    if isinstance(type_expr, (tuple, list)):
        checked = []
        for ty in type_expr:
            check_i = builtins_isinstance_1(span, container_expr, ty)
            checked.append(check_i)
        return any(span, *checked)
    else:
        return builtins_isinstance_1(span, container_expr, type_expr)


def builtins_exception(span, exc_cls_name, *args, **kwargs):
    if len(kwargs) != 0:
        raise TypeError(f'{exc_cls_name}() takes no keyword arguments')
    return call_extern(_type.ExceptionType(exc_cls_name), "MAKE_PY_" + exc_cls_name, span, *args)


def nd_module_add(span, lhs, rhs):
    func_name = "ir.nd_module_add"
    return hlo_call_intrin(_type.NDArrayType(), func_name, span, lhs, rhs)


def nd_module_sub(span, lhs, rhs):
    func_name = "ir.nd_module_sub"
    return hlo_call_intrin(_type.NDArrayType(), func_name, span, lhs, rhs)


def nd_module_div(span, lhs, rhs):
    func_name = "ir.nd_module_div"
    return hlo_call_intrin(_type.NDArrayType(), func_name, span, lhs, rhs)


def nd_module_mul(span, lhs, rhs):
    func_name = "ir.nd_module_mul"
    return hlo_call_intrin(_type.NDArrayType(), func_name, span, lhs, rhs)


def nd_module_rand(span, shape):
    func_name = "ir.nd_module_rand"
    return hlo_call_intrin(_type.NDArrayType(), func_name, span, shape)


def nd_module_concatenate(span, seq, *args):
    func_name = "ir.nd_module_concatenate"
    return hlo_call_intrin(_type.NDArrayType(), func_name, span, seq, *args)


def nd_module_stack(span, seq, *args):
    func_name = "ir.nd_module_stack"
    return hlo_call_intrin(_type.NDArrayType(), func_name, span, seq, *args)


def list_module_sort(span, seq, *args):
    func_name = "ir.list_module_sort"
    return hlo_call_intrin(_type.VoidType(), func_name, span, seq, *args)


def list_module_nth_element(span, seq, *args):
    func_name = "ir.list_module_nth_element"
    return hlo_call_intrin(_type.VoidType(), func_name, span, seq, *args)


def list_module_heapify(span, seq, *args):
    func_name = "ir.list_module_heapify"
    return hlo_call_intrin(_type.VoidType(), func_name, span, seq, *args)


def list_module_heap_replace(span, seq, *args):
    func_name = "ir.list_module_heap_replace"
    return hlo_call_intrin(_type.VoidType(), func_name, span, seq, *args)


def list_module_heap_pushpop(span, seq, *args):
    func_name = "ir.list_module_heap_pushpop"
    return hlo_call_intrin(_type.ObjectType(), func_name, span, seq, *args)


def cuda_module_default_stream(span, device_id):
    func_name = "ir.cuda_module_default_stream"
    return hlo_call_intrin(_type.OpaqueObjectType(), func_name, span, device_id)


def cuda_module_create_stream(span, device_id):
    func_name = "ir.cuda_module_create_stream"
    return hlo_call_intrin(_type.OpaqueObjectType(), func_name, span, device_id)


def cuda_module_stream_sync(span, stream, device_id):
    func_name = "ir.cuda_module_stream_sync"
    return hlo_call_intrin(_type.VoidType(), func_name, span, stream, device_id)


def pickle_serialize(span, o):
    func_name = "ir.pickle_serialize"
    return hlo_call_intrin(_type.UnicodeType(), func_name, span, o)


def pickle_deserialize(span, s):
    func_name = "ir.pickle_deserialize"
    return hlo_call_intrin(_type.ObjectType(), func_name, span, s)


def numpy_ops(span, attr, *args):
    func_name = "ir.numpy_ops"
    if isinstance(attr, str):
        attr = StringImm(attr, span=span)
    elif isinstance(attr, (bytes, bytearray)):
        attr = StringImm(attr.decode(), span=span)
    return hlo_call_intrin(_type.ObjectType(), func_name, span, attr, *args)


def torch_ops(span, attr, *args):
    func_name = "ir.torch_ops"
    if isinstance(attr, str):
        attr = StringImm(attr, span=span)
    elif isinstance(attr, (bytes, bytearray)):
        attr = StringImm(attr.decode(), span=span)
    return hlo_call_intrin(_type.ObjectType(), func_name, span, attr, *args)


def make_kwargs_op(span, **kwargs):
    func_name = "ir.make_kwargs_op"
    args = []
    for key, value in kwargs.items():
        args.append(StringImm(key))
        args.append(value)
    return hlo_call_intrin(_type.ObjectType(), func_name, span, *args)
