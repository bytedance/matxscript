# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The TypeInfer is inspired by incubator-tvm.
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
"""Type relation and function for type checking."""
import typing
from matx.ir.base import BaseExpr
from .. import _ffi
from ..runtime import List, Dict, Set
from .. import ir

from .type import Type, PrimType
from .type import ListType, DictType, SetType, ObjectType
from .type import StringType, UnicodeType, FileType, TupleType, UserDataType
from .adt import ClassType
from . import _ffi_api
from ._converter import to_ir_object as _to_ir
from .expr import NoneExpr


def lift_type(ty_1: Type, ty_2: Type):
    return _ffi_api.InferLiftType(ty_1, ty_2)


def remove_view(t: Type):
    if isinstance(t, StringType) and t.is_view:
        return StringType(is_view=False)
    if isinstance(t, UnicodeType) and t.is_view:
        return UnicodeType(is_view=False)
    if isinstance(t, ObjectType) and t.is_view:
        return ObjectType(is_view=False)
    return t


def type_convertible(from_ty: Type, to_ty: Type):
    return _ffi_api.IsTypeConvertible(from_ty, to_ty)


def __is_base_of(base: Type, derived: Type, allow_same: bool):
    return _ffi_api.IsBaseTypeOf(base, derived, allow_same)


def smart_adapt_to(value: BaseExpr, to: Type, span=None):
    if (__is_base_of(value.checked_type, to, False)
            or __is_base_of(to, value.checked_type, False)):
        return ir.HLOCast(to, value, span)
    if isinstance(value.checked_type, ObjectType) and not isinstance(to, ObjectType):
        return ir.HLOCast(to, value, span)
    if isinstance(to, ObjectType) and not isinstance(value.checked_type, ObjectType):
        return ir.HLOCast(to, value, span)
    if isinstance(to, StringType) and isinstance(value.checked_type, StringType):
        if to.is_view ^ value.checked_type.is_view:
            return ir.HLOCast(to, value, span)
    if isinstance(to, UnicodeType) and isinstance(value.checked_type, UnicodeType):
        if to.is_view ^ value.checked_type.is_view:
            return ir.HLOCast(to, value, span)
    return value


def type_inference(value):
    if hasattr(value, 'dtype'):
        ret_type = PrimType(str(value.dtype))
    elif hasattr(value, 'ret_type'):
        ret_type = value.ret_type
    elif isinstance(value, bool):
        ret_type = PrimType('bool')
    elif isinstance(value, float):
        ret_type = PrimType('float64')
    elif isinstance(value, int):
        ret_type = PrimType('int64')
    elif isinstance(value, List):
        ret_type = ListType()
    elif isinstance(value, Dict):
        ret_type = DictType()
    elif isinstance(value, Set):
        ret_type = SetType()
    elif isinstance(value, (bytes, bytearray, ir.StringImm)):
        ret_type = StringType()
    elif isinstance(value, (str, ir.UnicodeImm)):
        ret_type = UnicodeType()
    elif hasattr(value, 'checked_type'):
        ret_type = value.checked_type
    else:
        raise NotImplementedError('Cannot automatically inference the type.'
                                  ' value={}'.format(value))
    return ret_type


def is_type_of(__obj: BaseExpr, __class_or_tuple: typing.Union[type, typing.Tuple[type, ...]]):
    if isinstance(__class_or_tuple, tuple):
        return any(is_type_of(__obj, __class) for __class in __class_or_tuple)
    if not issubclass(__class_or_tuple, Type):
        return False
    if __class_or_tuple is PrimType.IntType:
        return is_type_of(__obj, PrimType) and __obj.checked_type.dtype in ('int32', 'int64')
    if __class_or_tuple is PrimType.FloatType:
        return is_type_of(
            __obj, PrimType) and __obj.checked_type.dtype in (
            'float32', 'float64')
    if __class_or_tuple is PrimType.BoolType:
        return is_type_of(__obj, PrimType) and __obj.checked_type.dtype in ('bool',)
    return isinstance(__obj.checked_type, __class_or_tuple)


def slice_type_inference(container_type):
    if isinstance(container_type, ListType):
        ret_type = container_type.item_type
    else:
        raise NotImplementedError('Cannot automatically inference the type.'
                                  ' value={}'.format(container_type))
    return ret_type


def infer_iterator_value_type(container_type):
    return _ffi_api.InferIteratorValueType(container_type)


def infer_nth_item_type(container_type, index):
    return _ffi_api.InferNthItemType(container_type, index)
