# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the expressions is inspired by TVM.
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
# pylint: disable=redefined-builtin
"""IR expression nodes.

Each expression node have subfields that can be visited from python side.
For example, you can use addexp.a to get the left operand of an Add node.

.. code-block:: python

  x = ir.PrimVar("n", "int32")
  y = x + 2
  assert(isinstance(y, ir.PrimAdd))
  assert(y.a == x)
"""
from .. import _ffi

from ..runtime import Object, ObjectGeneric, DataType, DataTypeCode
from ._converter import const
from .base import BaseExpr, PrimExpr, HLOExpr, Span
from .op_expr import Op
from . import generic as _generic
from . import _ffi_api
from . import type as _type
from ._converter import to_ir_object as _to_ir


def _assert_is_prim_expr(value):
    msg = "Expect PrimExpr, but received : {0}".format(type(value))
    assert isinstance(value, PrimExpr), msg
    return value


def _cast_to_prim_expr(value):
    if isinstance(value, PrimExpr):
        return value
    hlo_type = value.checked_type
    assert isinstance(hlo_type, _type.PrimType)
    return HLOCastPrim(hlo_type.dtype, value)


def div_ambiguity_error():
    return RuntimeError(
        "MATX supports multiple types of integer divisions, "
        + "please call div, indexdiv/indexmod, floordiv/floormod "
        + " or truncdiv/truncmod directly to avoid ambiguity in the code."
    )


def _dtype_is_int(value):
    if isinstance(value, int):
        return True
    return isinstance(value, ExprOp) and DataType(value.dtype).type_code == DataTypeCode.INT


def _dtype_is_float(value):
    if isinstance(value, float):
        return True
    return isinstance(value, ExprOp) and DataType(value.dtype).type_code == DataTypeCode.FLOAT


class ExprOp(object):
    """Operator overloading for Expr like expressions."""

    def __add__(self, other):
        return _generic.add(self, other)

    def __radd__(self, other):
        return _generic.add(other, self)

    def __sub__(self, other):
        return _generic.subtract(self, other)

    def __rsub__(self, other):
        return _generic.subtract(other, self)

    def __mul__(self, other):
        return _generic.multiply(self, other)

    def __rmul__(self, other):
        return _generic.multiply(other, self)

    def __div__(self, other):
        return _generic.divide(self, other)

    def __rdiv__(self, other):
        return _generic.divide(other, self)

    def __truediv__(self, other):
        return _generic.divide(self, other)

    def __rtruediv__(self, other):
        return _generic.divide(other, self)

    def __floordiv__(self, other):
        return _generic.floordiv(self, other)

    def __rfloordiv__(self, other):
        return _generic.floordiv(other, self)

    def __mod__(self, other):
        return _generic.floormod(self, other)

    def __rmod__(self, other):
        return _generic.floormod(other, self)

    def __neg__(self):
        neg_one = const(-1, self.dtype)
        return self.__mul__(neg_one)

    def __lshift__(self, other):
        return _generic.left_shift(self, other)

    def __rlshift__(self, other):
        return _generic.left_shift(other, self)

    def __rshift__(self, other):
        return _generic.right_shift(self, other)

    def __rrshift__(self, other):
        return _generic.right_shift(other, self)

    def __and__(self, other):
        return _generic.bitwise_and(self, other)

    def __rand__(self, other):
        return _generic.bitwise_and(other, self)

    def __or__(self, other):
        return _generic.bitwise_or(self, other)

    def __ror__(self, other):
        return _generic.bitwise_or(other, self)

    def __xor__(self, other):
        return _generic.bitwise_xor(self, other)

    def __rxor__(self, other):
        return _generic.bitwise_xor(other, self)

    def __invert__(self):
        return _generic.bitwise_not(self)

    def __lt__(self, other):
        return _generic.less_than(self, other)

    def __le__(self, other):
        return _generic.less_or_equal(self, other)

    def __eq__(self, other):
        return _generic.equal(self, other)

    def __ne__(self, other):
        return _generic.notequal(self, other)

    def __gt__(self, other):
        return _generic.greater_than(self, other)

    def __ge__(self, other):
        return _generic.greater_or_equal(self, other)

    def __nonzero__(self):
        raise ValueError(
            "Cannot use and / or / not operator to Expr, hint: "
            + "use ir.all / ir.any instead"
        )

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other):
        """Build an equal check expression with other expr.

        Parameters
        ----------
        other : PrimExpr
            The other expression

        Returns
        -------
        ret : PrimExpr
            The equality expression.
        """
        return _ffi_api._OpEQ(self, other)

    def astype(self, dtype):
        """Cast the expression to other type.

        Parameters
        ----------
        dtype : str
            The type of new expression

        Returns
        -------
        expr : PrimExpr
            Expression with new type
        """
        return _generic.cast(self, _to_ir(dtype))


class EqualOp(ObjectGeneric, ExprOp):
    """Deferred equal operator.

    This is used to support sugar that a == b can either
    mean Object.same_as or Object.equal.

    Parameters
    ----------
    a : PrimExpr
        Left operand.

    b : PrimExpr
        Right operand.
    """

    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __nonzero__(self):
        return self.a.same_as(self.b)

    def __bool__(self):
        return self.__nonzero__()

    def asobject(self):
        """Convert object."""
        return _ffi_api._OpEQ(self.a, self.b)


class NotEqualOp(ObjectGeneric, ExprOp):
    """Deferred NE operator.

    This is used to support sugar that a != b can either
    mean not Object.same_as or make.NE.

    Parameters
    ----------
    a : PrimExpr
        Left operand.

    b : PrimExpr
        Right operand.
    """

    # This class is not manipulated by C++. So use python's identity check function is sufficient
    same_as = object.__eq__

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __nonzero__(self):
        return not self.a.same_as(self.b)

    def __bool__(self):
        return self.__nonzero__()

    def asobject(self):
        """Convert object."""
        return _ffi_api._OpNE(self.a, self.b)


class IntImmEnum(ObjectGeneric):
    """Lazily evaluate an IntImm in case
    the constructor is not available in runtime.

    Parameters
    ----------
    value : int
        The enum value
    """

    def __init__(self, value):
        self.value = value

    def asobject(self):
        """Convert object."""
        return IntImm("int32", self.value)


class PrimExprWithOp(ExprOp, PrimExpr):
    """Helper base class to inherit from PrimExpr."""

    # In Python3, We have to explicitly tell interpreter to retain __hash__ if we overide __eq__
    # https://docs.python.org/3.1/reference/datamodel.html#object.__hash__
    __hash__ = PrimExpr.__hash__


class BinaryOpExpr(PrimExprWithOp):
    pass


class CmpExpr(PrimExprWithOp):
    pass


class LogicalExpr(PrimExprWithOp):
    pass


@_ffi.register_object("ir.PrimVar")
class PrimVar(PrimExprWithOp):
    """Symbolic variable.

    Parameters
    ----------
    name : str
        The name

    dtype : Union[str, irType]
        The data type
    """

    def __init__(self, name, dtype, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimVar, _to_ir(name), _to_ir(dtype), span)


@_ffi.register_object
class FloatImm(PrimExprWithOp):
    """Float constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : float
        The constant value.
    """

    def __init__(self, dtype, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.FloatImm, _to_ir(dtype), value, span)


@_ffi.register_object
class IntImm(PrimExprWithOp):
    """Int constant.

    Parameters
    ----------
    dtype : str
        The data type

    value : int
        The constant value.
    """

    def __init__(self, dtype, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.IntImm, _to_ir(dtype), value, span)

    def __hash__(self):
        return self.value

    def __int__(self):
        return self.value

    def __nonzero__(self):
        return self.value != 0

    def __eq__(self, other):
        return _ffi_api._OpEQ(self, other)

    def __ne__(self, other):
        return _ffi_api._OpNE(self, other)

    def __bool__(self):
        return self.__nonzero__()


@_ffi.register_object("ir.PrimCast")
class PrimCast(PrimExprWithOp):
    """Cast expression.

    Parameters
    ----------
    dtype : str
        The data type

    value : PrimExpr
        The value of the function.
    """

    def __init__(self, dtype, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimCast, _to_ir(dtype), _to_ir(value), span)


@_ffi.register_object("ir.PrimAdd")
class PrimAdd(BinaryOpExpr):
    """Add node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimAdd, a, b, span)


@_ffi.register_object("ir.PrimSub")
class PrimSub(BinaryOpExpr):
    """Sub node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimSub, a, b, span)


@_ffi.register_object("ir.PrimMul")
class PrimMul(BinaryOpExpr):
    """Mul node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimMul, a, b, span)


@_ffi.register_object("ir.PrimDiv")
class PrimDiv(BinaryOpExpr):
    """Div node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimDiv, a, b, span)


@_ffi.register_object("ir.PrimMod")
class PrimMod(BinaryOpExpr):
    """Mod node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimMod, a, b, span)


@_ffi.register_object("ir.PrimFloorDiv")
class PrimFloorDiv(BinaryOpExpr):
    """FloorDiv node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimFloorDiv, a, b, span)


@_ffi.register_object("ir.PrimFloorMod")
class PrimFloorMod(BinaryOpExpr):
    """FloorMod node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimFloorMod, a, b, span)


@_ffi.register_object("ir.PrimMin")
class PrimMin(BinaryOpExpr):
    """Min node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimMin, a, b, span)


@_ffi.register_object("ir.PrimMax")
class PrimMax(BinaryOpExpr):
    """Max node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimMax, a, b, span)


@_ffi.register_object("ir.PrimEQ")
class PrimEQ(CmpExpr):
    """EQ node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimEQ, a, b, span)


@_ffi.register_object("ir.PrimNE")
class PrimNE(CmpExpr):
    """NE node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimNE, a, b, span)


@_ffi.register_object("ir.PrimLT")
class PrimLT(CmpExpr):
    """LT node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimLT, a, b, span)


@_ffi.register_object("ir.PrimLE")
class PrimLE(CmpExpr):
    """LE node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimLE, a, b, span)


@_ffi.register_object("ir.PrimGT")
class PrimGT(CmpExpr):
    """GT node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimGT, a, b, span)


@_ffi.register_object("ir.PrimGE")
class PrimGE(CmpExpr):
    """GE node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimGE, a, b, span)


@_ffi.register_object("ir.PrimAnd")
class PrimAnd(LogicalExpr):
    """And node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimAnd, a, b, span)


@_ffi.register_object("ir.PrimOr")
class PrimOr(LogicalExpr):
    """Or node.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand.

    b : PrimExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimOr, a, b, span)


@_ffi.register_object("ir.PrimNot")
class PrimNot(LogicalExpr):
    """Not node.

    Parameters
    ----------
    a : PrimExpr
        The input value
    """

    def __init__(self, a, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimNot, a, span)


@_ffi.register_object("ir.PrimSelect")
class PrimSelect(PrimExprWithOp):
    """Select node.

    Note
    ----
    Select may compute both true_value and false_value.
    Use :py:class:`ir.if_then_else` instead if you want to
    get a conditional expression that only evaluates
    the correct branch.

    Parameters
    ----------
    condition : PrimExpr
        The condition expression.

    true_value : PrimExpr
        The value to take when condition is true.

    false_value : PrimExpr
        The value to take when condition is false.

    """

    def __init__(self, condition, true_value, false_value, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.PrimSelect, condition, true_value, false_value, span)


class CallEffectKind:
    """Possible kinds of Call effects."""

    # only expose up to opaque
    ExprAnnotation = IntImmEnum(0)
    Pure = IntImmEnum(1)
    ReadState = IntImmEnum(2)
    UpdateState = IntImmEnum(3)
    Opaque = UpdateState


@_ffi.register_object("ir.PrimCall")
class PrimCall(PrimExprWithOp):
    """Call node.

    Parameters
    ----------
    dtype : str
        The return data type

    op : Union[HLOExpr, str]
        The function to be called, or the name
        to the global Op

    args : list of Expr
        The input arguments to the call
    """

    def __init__(self, dtype, op, args, span=Span()):
        if isinstance(op, str):
            if not op.startswith("ir."):
                raise ValueError(
                    ("Cannot handle str op argument %s. This function only handles str "
                     + "argument with the ir namespace. If you are "
                     + "certain about the intrinsic name, pass in Op.get(name) instead"
                     )
                    % op
                )
            op = Op.get(op)
        self.__init_handle_by_constructor__(_ffi_api.PrimCall,
                                            _to_ir(dtype),
                                            _to_ir(op),
                                            _to_ir(args),
                                            span)


@_ffi.register_object("ir.PrimLet")
class PrimLet(PrimExprWithOp):
    """Let node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : PrimExpr
        The value in to be binded.

    body : PrimExpr
        The body expression.
    """

    def __init__(self, var, value, body, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.PrimLet, var, value, body, span)


###############################################################################
# Base node of all non-primitive expressions.
###############################################################################

class HLOExprWithOp(ExprOp, HLOExpr):
    """Basetype of all high level expressions that defines op overloading."""

    __hash__ = HLOExpr.__hash__


class HLOBinaryOpExpr(HLOExprWithOp):
    pass


class HLOCmpExpr(HLOExprWithOp):
    pass


class HLOLogicalExpr(HLOExprWithOp):
    pass


@_ffi.register_object("ir.HLOAdd")
class HLOAdd(HLOBinaryOpExpr):
    """HLOAdd node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOAdd, a, b, span)


@_ffi.register_object("ir.HLOSub")
class HLOSub(HLOBinaryOpExpr):
    """Sub node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOSub, a, b, span)


@_ffi.register_object("ir.HLOMul")
class HLOMul(HLOBinaryOpExpr):
    """Mul node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOMul, a, b, span)


@_ffi.register_object("ir.HLOFloorDiv")
class HLOFloorDiv(HLOBinaryOpExpr):
    """FloorDiv node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOFloorDiv, a, b, span)


@_ffi.register_object("ir.HLOFloorMod")
class HLOFloorMod(HLOBinaryOpExpr):
    """FloorMod node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOFloorMod, a, b, span)


@_ffi.register_object("ir.HLOEqual")
class HLOEqual(HLOCmpExpr):
    """EQ node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOEqual, a, b, span)


@_ffi.register_object("ir.HLONotEqual")
class HLONotEqual(HLOCmpExpr):
    """NE node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLONotEqual, a, b, span)


@_ffi.register_object("ir.HLOLessThan")
class HLOLessThan(HLOCmpExpr):
    """LT node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOLessThan, a, b, span)


@_ffi.register_object("ir.HLOLessEqual")
class HLOLessEqual(HLOCmpExpr):
    """LE node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOLessEqual, a, b, span)


@_ffi.register_object("ir.HLOGreaterThan")
class HLOGreaterThan(HLOCmpExpr):
    """GT node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOGreaterThan, a, b, span)


@_ffi.register_object("ir.HLOGreaterEqual")
class HLOGreaterEqual(HLOCmpExpr):
    """GE node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOGreaterEqual, a, b, span)


@_ffi.register_object("ir.HLOAnd")
class HLOAnd(HLOLogicalExpr):
    """And node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOAnd, a, b, span)


@_ffi.register_object("ir.HLOOr")
class HLOOr(HLOLogicalExpr):
    """Or node.

    Parameters
    ----------
    a : BaseExpr
        The left hand operand.

    b : BaseExpr
        The right hand operand.
    """

    def __init__(self, a, b, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOOr, a, b, span)


@_ffi.register_object("ir.HLONot")
class HLONot(HLOLogicalExpr):
    """Not node.

    Parameters
    ----------
    a : BaseExpr
        The input value
    """

    def __init__(self, a, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLONot, a, span)


@_ffi.register_object("ir.HLOCastPrim")
class HLOCastPrim(HLOExprWithOp):
    """Cast expression.

    Parameters
    ----------
    dtype : str
        The data type

    value : BaseExpr
        The value of the function.
    """

    def __init__(self, dtype, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOCastPrim, _to_ir(dtype), value, span)


@_ffi.register_object("ir.HLOCast")
class HLOCast(HLOExprWithOp):
    """Cast expression.

    Parameters
    ----------
    ty : Type
        The data type

    value : BaseExpr
        The value of the function.
    """

    def __init__(self, ty, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOCast, ty, value, span)


@_ffi.register_object("ir.InitializerList")
class InitializerList(HLOExprWithOp):
    """InitializerList expression that groups several fields together.

    Parameters
    ----------
    fields : List[ir.BaseExpr]
        The fields in the InitializerList.
    """

    def __init__(self, fields, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.InitializerList, _to_ir(fields), span)

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on InitializerList")


@_ffi.register_object("ir.InitializerDict")
class InitializerDict(HLOExprWithOp):
    """InitializerList expression that groups several fields together.

    Parameters
    ----------
    fields : Map[ir.BaseExpr, ir.BaseExpr]
        The fields in the InitializerDict.
    """

    def __init__(self, fields, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.InitializerDict, _to_ir(fields), span)

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on InitializerList")


@_ffi.register_object("ir.EnumAttr")
class EnumAttr(HLOExprWithOp):
    """EnumAttr expression, only for codegen int32 enum value.

    Parameters
    ----------
    enum_str : str
        The enum value.
    """

    def __init__(self, enum_str, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.EnumAttr, _to_ir(enum_str), span)


@_ffi.register_object("ir.Call")
class Call(HLOExprWithOp):
    """Function call node in matx ir.

    Call node corresponds the operator application node
    in computational graph terminology.

    Parameters
    ----------
    ret_type: ir.Type

    op: ir.Op or any ir.Expr with function type.
        The operation to be called.

    args: List[ir.Expr]
        The arguments to the call.

    span: Span
        The source code info.

    type_args: Optional[List[Union[ir.Type, int]]]
        The additional type arguments, this is only
        used in advanced usecase of template functions.
    """

    def __init__(self, ret_type, op, args, span=Span(), type_args=None):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(_ffi_api.Call,
                                            _to_ir(ret_type),
                                            _to_ir(op),
                                            _to_ir(args),
                                            span,
                                            _to_ir(type_args))


@_ffi.register_object("GlobalVar")
class GlobalVar(HLOExpr):
    """A global variable in the IR.

    GlobalVar is used to refer to the global functions
    stored in the IRModule.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """

    def __init__(self, name_hint, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.GlobalVar, _to_ir(name_hint), span)

    def __call__(self, *args):
        """Call the global variable.

        Parameters
        ----------
        args: List[HLOExpr]
            The arguments to the call.

        Returns
        -------
        call: BaseExpr
            A call taking the variable as a function.
        """
        # pylint: disable=import-outside-toplevel
        if all(isinstance(x, HLOExpr) for x in args):
            return Call(self, args)
        arg_types = [type(x) for x in args]
        raise RuntimeError(
            "Do not know how to handle GlobalVar.__call__ for types {}".format(arg_types)
        )


@_ffi.register_object("ir.Tuple")
class Tuple(HLOExprWithOp):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : List[ir.BaseExpr]
        The fields in the tuple.
    """

    def __init__(self, fields, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.Tuple, _to_ir(fields), span)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


@_ffi.register_object("ir.HLOVar")
class HLOVar(HLOExprWithOp):
    """A local variable in high level expr.

    Local variable can be used to declare input
    arguments to a function, or intermediate variables.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: ir.Type, optional
        The type annotation on the variable.
    """

    def __init__(self, name_hint, type_annotation=None, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.HLOVar,
                                            _to_ir(name_hint),
                                            _to_ir(type_annotation),
                                            span)

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = str(self.vid.name_hint)
        return name


@_ffi.register_object("ir.ClassGetItem")
class ClassGetItem(HLOExprWithOp):
    """Get attrbute from a user class.

    Parameters
    ----------
    expr: ir.HLOExpr
        The input user class expression.

    attr: str
        The attribute name.
    """

    def __init__(self, expr, attr, span=Span()):
        if isinstance(attr, (str, bytes, bytearray)):
            attr = StringImm(attr)
        self.__init_handle_by_constructor__(_ffi_api.ClassGetItem, _to_ir(expr), _to_ir(attr), span)


@_ffi.register_object("ir.NoneExpr")
class NoneExpr(HLOExprWithOp):
    """NoneExpr.

    Parameters
    ----------
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.NoneExpr)


@_ffi.register_object("ir.StringImm")
class StringImm(HLOExprWithOp):
    """String constant.

    Parameters
    ----------
    value : str, bytes
        The value of the function.
    """

    def __init__(self, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.StringImm, _to_ir(value), span)

    def __eq__(self, other):
        if isinstance(other, ExprOp):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, ExprOp):
            return self.value != other.value
        return self.value != other

    __hash__ = PrimExpr.__hash__


@_ffi.register_object("ir.UnicodeImm")
class UnicodeImm(HLOExprWithOp):
    """String constant.

    Parameters
    ----------
    value : str
        The value of the function.
    """

    def __init__(self, value, span=Span()):
        self.__init_handle_by_constructor__(_ffi_api.UnicodeImm, _to_ir(value), span)

    def __eq__(self, other):
        if isinstance(other, ExprOp):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, ExprOp):
            return self.value != other.value
        return self.value != other

    __hash__ = PrimExpr.__hash__


@_ffi.register_object("ir.HLOEnumerate")
class HLOEnumerate(HLOExprWithOp):
    """enumerate expr.

    Parameters
    ----------
    value : HLOExpr
        The container.
    start : Union[PrimExpr, int]
        The start of enumerate.
    """

    def __init__(self, value, start=0, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.HLOEnumerate, _to_ir(value), _to_ir(start), span
        )


@_ffi.register_object("ir.HLOZip")
class HLOZip(HLOExprWithOp):
    """zip expr.

    Parameters
    ----------
    values : List[HLOExpr]
        The containers.
    """

    def __init__(self, values, span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.HLOZip, _to_ir(values), span
        )
