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
"""Common base structures."""
from .. import _ffi
from .. import runtime

from ..runtime import Object
from ._converter import to_ir_object as _to_ir
from . import _ffi_api
from . import _ffi_node_api


class Node(Object):
    """Base class of all IR Nodes, implements astext function."""

    def astext(self):
        """Get the text format of the expression.

        Parameters
        ----------

        Returns
        -------
        text : str
            The text format of the expression.

        Notes
        -----
        The meta data section is necessary to fully parse the text format.
        However, it can contain dumps that are big (e.g constant weights),
        so it can be helpful to skip printing the meta data section.
        """
        node_text = _ffi_api.AsText(self)
        if isinstance(node_text, (bytes, bytearray)):
            node_text = node_text.decode('utf-8')
        return node_text

    def __str__(self):
        node_text = _ffi_api.PrettyPrint(self)
        if isinstance(node_text, (bytes, bytearray)):
            node_text = node_text.decode('utf-8')
        return node_text


@_ffi.register_object("SourceName")
class SourceName(Object):
    """A identifier for a source location.

    Parameters
    ----------
    name : str
        The name of the source.
    """

    def __init__(self, name):
        self.__init_handle_by_constructor__(_ffi_api.SourceName, _to_ir(name))


@_ffi.register_object("Span")
class Span(Object):
    """Specifies a location in a source program.

    Parameters
    ----------
    source : SourceName
        The source name.

    lineno : int
        The line number.

    col_offset : int
        The column offset of the location.
    """

    def __init__(self, file_name='', lineno=-1, func_name='', source_code=''):
        self.__init_handle_by_constructor__(
            _ffi_api.Span,
            file_name,
            lineno,
            func_name,
            source_code
        )


def structural_equal(lhs, rhs, map_free_vars=False):
    """Check structural equality of lhs and rhs.

    The structural equality is recursively defined in the DAG of IRNodes.
    There are two kinds of nodes:

    - Graph node: a graph node in lhs can only be mapped as equal to
      one and only one graph node in rhs.
    - Normal node: equality is recursively defined without the restriction
      of graph nodes.

    Vars(tir::Var, TypeVar) and non-constant relay expression nodes are graph nodes.
    For example, it means that `%1 = %x + %y; %1 + %1` is not structurally equal
    to `%1 = %x + %y; %2 = %x + %y; %1 + %2` in relay.

    A var-type node(e.g. tir::Var, TypeVar) can be mapped as equal to another var
    with the same type if one of the following condition holds:

    - They appear in a same definition point(e.g. function argument).
    - They points to the same VarNode via the same_as relation.
    - They appear in a same usage point, and map_free_vars is set to be True.

    The rules for var are used to remap variables occurs in function
    arguments and let-bindings.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    map_free_vars : bool
        Whether or not shall we map free vars that does
        not bound to any definitions as equal to each other.

    Return
    ------
    result : bool
        The comparison result.

    See Also
    --------
    structural_hash
    assert_strucural_equal
    """
    lhs = _to_ir(lhs)
    rhs = _to_ir(rhs)
    return bool(_ffi_node_api.structrual_equal(lhs, rhs, False, map_free_vars))


def get_first_structural_mismatch(lhs, rhs, map_free_vars=False):
    """Like structural_equal(), but returns the ObjectPaths of the first detected mismatch.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    map_free_vars : bool
        Whether free variables (i.e. variables without a definition site) should be mapped
        as equal to each other.

    Returns
    -------
    mismatch: Optional[Tuple[ObjectPath, ObjectPath]]
        `None` if `lhs` and `rhs` are structurally equal.
        Otherwise, a tuple of two ObjectPath objects that point to the first detected mismtach.
    """
    lhs = _to_ir(lhs)
    rhs = _to_ir(rhs)
    mismatch = _ffi_node_api.GetFirstStructuralMismatch(lhs, rhs, map_free_vars)
    if mismatch is None:
        return None
    else:
        return mismatch.lhs_path, mismatch.rhs_path


def assert_structural_equal(lhs, rhs, map_free_vars=False):
    """Assert lhs and rhs are structurally equal to each other.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    map_free_vars : bool
        Whether or not shall we map free vars that does
        not bound to any definitions as equal to each other.

    Raises
    ------
    ValueError : if assertion does not hold.

    See Also
    --------
    structural_equal
    """
    lhs = _to_ir(lhs)
    rhs = _to_ir(rhs)
    _ffi_node_api.structrual_equal(lhs, rhs, True, map_free_vars)


def structural_hash(node, map_free_vars=False):
    """Compute structural hash of node

    The structural hash value is recursively defined in the DAG of IRNodes.
    There are two kinds of nodes:

    - Normal node: the hash value is defined by its content and type only.
    - Graph node: each graph node will be assigned a unique index ordered by the
      first occurence during the visit. The hash value of a graph node is
      combined from the hash values of its contents and the index.

    structural_hash is made to be concistent with structural_equal.
    If two nodes are structurally equal to each other,
    then their structural hash (with the same map_free_vars option)
    should be equal to each other as well.

    If the structural hash of two nodes equals to each other,
    then it is highly likely(except for rare hash value collison cases)
    that the two nodes are structurally equal to each other.

    Parameters
    ----------
    node : Object
        The input to be hashed.

    map_free_vars : bool
        If map_free_vars is set to true, we will hash free variables
        by the order of their occurences. Otherwise, we will hash by
        their in-memory pointer address.

    Return
    ------
    result : int
        The hash result

    See Also
    --------
    structrual_equal
    """
    return _ffi_node_api.structural_hash(node, map_free_vars)


class BaseExpr(Node):
    """Base class of all the expressions."""

    @property
    def checked_type(self):
        """Get the checked type of matx.ir.HLOExpr.

        Returns
        -------
        checked_type : matx.ir.Type
            The checked type.
        """
        ret = self._checked_type_
        if ret is None:
            raise ValueError("The type checker has not populated" " the checked_type for this node")
        return ret

    def py_type_name(self):
        """only for error reporter
        """
        return self.checked_type.get_py_type_name()


class PrimExpr(BaseExpr):
    """Base class of all primitive expressions.

    PrimExpr is used in the low-level code
    optimizations and integer analysis.
    """


class HLOExpr(BaseExpr):
    """Base class of all non-primitive expressions."""
