# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the Type is inspired by TVM.
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
"""Unified type system in the project."""
from enum import IntEnum

from .. import _ffi
from .. import ir

from .base import Node
from . import _ffi_api
from ._converter import to_ir_object as _to_ir


class Type(Node):
    """The base class of all types."""

    def __eq__(self, other):
        """Compare two types for structural equivalence."""
        return bool(ir.structural_equal(self, other))

    def __hash__(self):
        """Compare two types for structural equivalence."""
        return ir.structural_hash(self)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_runtime_type_code(self):
        """Get RTValue type code

        Returns
        -------
        code : int32_t
            RTValue type code, if unknown return INT16_MIN(-32768)

        """
        return _ffi_api.Type_GetRuntimeTypeCode(self)

    def get_py_type_name(self):
        return _ffi_api.Type_GetPythonTypeName(self)

    def py_type_name(self):
        return _ffi_api.Type_GetPythonTypeName(self)

    def is_full_typed(self):
        return bool(_ffi_api.Type_IsFullTyped(self))

    def is_iterable(self):
        return bool(_ffi_api.Type_IsIterable(self))


class TypeKind(IntEnum):
    """Possible kinds of TypeVars."""

    Type = 0
    ShapeVar = 1
    BaseType = 2
    Constraint = 4
    AdtHandle = 5


@_ffi.register_object("PrimType")
class PrimType(Type):
    """Primitive data type in the low level IR

    Parameters
    ----------
    dtype : str
        The runtime data type relates to the primtype.
    """

    class IntType(Type):
        pass

    class FloatType(Type):
        pass

    class BoolType(Type):
        pass
    # sub classes above are used for is_type_of check

    def __init__(self, dtype):
        self.__init_handle_by_constructor__(_ffi_api.PrimType, _to_ir(dtype))

    @property
    def dtype(self):
        return _ffi_api.PrimType_GetDType(self)


class VoidType(Type):
    """Primitive void type in the low level IR
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.VoidType)


@_ffi.register_object("PointerType")
class PointerType(Type):
    """PointerType used in the low-level TIR.

    Parameters
    ----------
    element_type : ir.Type
        The type of pointer's element.
    """

    def __init__(self, element_type):
        self.__init_handle_by_constructor__(_ffi_api.PointerType, element_type)


@_ffi.register_object("TypeVar")
class TypeVar(Type):
    """Type parameter in functions.

    A type variable represents a type placeholder which will
    be filled in later on. This allows the user to write
    functions which are generic over types.

    Parameters
    ----------
    name_hint: str
        The name of the type variable. This name only acts as a hint, and
        is not used for equality.

    kind : Optional[TypeKind]
        The kind of the type parameter.
    """

    def __init__(self, name_hint, kind=TypeKind.Type):
        self.__init_handle_by_constructor__(_ffi_api.TypeVar, name_hint, kind)

    def __call__(self, *args):
        """Create a type call from this type.

        Parameters
        ----------
        args: List[Type]
            The arguments to the type call.

        Returns
        -------
        call: Type
            The result type call.
        """
        # pylint: disable=import-outside-toplevel
        from .type_relation import TypeCall

        return TypeCall(self, args)


@_ffi.register_object("GlobalTypeVar")
class GlobalTypeVar(Type):
    """A global type variable that is used for defining new types or type aliases.

    Parameters
    ----------
    name_hint: str
        The name of the type variable. This name only acts as a hint, and
        is not used for equality.

    kind : Optional[TypeKind]
        The kind of the type parameter.
    """

    def __init__(self, name_hint, kind=TypeKind.AdtHandle):
        self.__init_handle_by_constructor__(_ffi_api.GlobalTypeVar, _to_ir(name_hint), kind)

    def __call__(self, *args):
        """Create a type call from this type.

        Parameters
        ----------
        args: List[Type]
            The arguments to the type call.

        Returns
        -------
        call: Type
            The result type call.
        """
        # pylint: disable=import-outside-toplevel
        from .type_relation import TypeCall

        return TypeCall(self, args)


@_ffi.register_object("TupleType")
class TupleType(Type):
    """The type of tuple values.

    Parameters
    ----------
    fields : List[Type]
        The fields in the tuple
    """

    def __init__(self, fields):
        self.__init_handle_by_constructor__(_ffi_api.TupleType, _to_ir(fields))

    def __getitem__(self, item):
        assert isinstance(item, int)
        return _ffi_api.TupleType_GetItem(self, item)

    def __len__(self):
        return _ffi_api.TupleType_Len(self)

    def aspylist(self):
        r = []
        for i in range(len(self)):
            r.append(self[i])
        return r


@_ffi.register_object("FuncType")
class FuncType(Type):
    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, and a return type.

    We can informally write them as:
    `forall (type_params), (arg_types) -> ret_type where type_constraints`

    Parameters
    ----------
    arg_types : List[relay.Type]
        The argument types

    ret_type : relay.Type
        The return type.

    type_params : Optional[List[relay.TypeVar]]
        The type parameters

    type_constraints : Optional[List[relay.TypeConstraint]]
        The type constraints.
    """

    def __init__(self, arg_types, ret_type, type_params=None, type_constraints=None):
        if type_params is None:
            type_params = []
        if type_constraints is None:
            type_constraints = []
        self.__init_handle_by_constructor__(
            _ffi_api.FuncType,
            _to_ir(arg_types),
            _to_ir(ret_type),
            _to_ir(type_params),
            _to_ir(type_constraints)
        )


@_ffi.register_object("ObjectType")
class ObjectType(Type):
    """Object Type in matx ir.

    """

    def __init__(self, is_view: bool = False):
        self.__init_handle_by_constructor__(_ffi_api.ObjectType, is_view)


@_ffi.register_object("StringType")
class StringType(Type):
    """String Type in matx ir.

    """

    def __init__(self, is_view: bool = False):
        self.__init_handle_by_constructor__(_ffi_api.StringType, is_view)


@_ffi.register_object("UnicodeType")
class UnicodeType(Type):
    """Unicode Type in matx ir.

    """

    def __init__(self, is_view: bool = False) -> None:
        self.__init_handle_by_constructor__(_ffi_api.UnicodeType, is_view)


@_ffi.register_object("ListType")
class ListType(Type):
    """List Type in matx ir.

    """

    def __init__(self, is_full_typed=False, item_type=ObjectType()):
        self.__init_handle_by_constructor__(_ffi_api.ListType, is_full_typed, item_type)

    @property
    def item_type(self):
        return _ffi_api.ListTypeGetItemType(self)


@_ffi.register_object("DictType")
class DictType(Type):
    """Dict Type in matx ir.

    """

    def __init__(self, is_full_typed=False, key_type=ObjectType(), value_type=ObjectType()):
        self.__init_handle_by_constructor__(_ffi_api.DictType, is_full_typed, key_type, value_type)


@_ffi.register_object("SetType")
class SetType(Type):
    """Set Type in matx ir.

    """

    def __init__(self, is_full_typed=False, item_type=ObjectType()):
        self.__init_handle_by_constructor__(_ffi_api.SetType, is_full_typed, item_type)


@_ffi.register_object("IteratorType")
class IteratorType(Type):
    """Iterator Type in matx ir.

    """

    def __init__(self, iterable_object_ty):
        self.__init_handle_by_constructor__(_ffi_api.IteratorType, iterable_object_ty)


@_ffi.register_object("FileType")
class FileType(Type):
    """File Type in matx ir.

    """

    def __init__(self, binary_mode=False):
        self.__init_handle_by_constructor__(_ffi_api.FileType, binary_mode)


@_ffi.register_object("TrieType")
class TrieType(Type):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TrieType)


@_ffi.register_object("UserDataType")
class UserDataType(Type):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.UserDataType)


@_ffi.register_object("NDArrayType")
class NDArrayType(Type):
    def __init__(self, ndim=-1, dtype=None):
        self.__init_handle_by_constructor__(_ffi_api.NDArrayType, ndim, dtype)


@_ffi.register_object("RegexType")
class RegexType(Type):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.RegexType)


@_ffi.register_object("OpaqueObjectType")
class OpaqueObjectType(Type):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.OpaqueObjectType)


@_ffi.register_object("ExceptionType")
class ExceptionType(Type):
    """A global type variable that is used for defining new types or type aliases.

    Parameters
    ----------
    name: str
        The name of the type. This name is used for equality.
    """

    def __init__(self, name):
        self.__init_handle_by_constructor__(_ffi_api.ExceptionType, _to_ir(name))
