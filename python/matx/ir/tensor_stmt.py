# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement:
# The structure of the Tensor Stmt is inspired by TensorIR(TVM).
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
"""Tensor Statement Node in MATX."""
from typing import Union, List, Mapping, Optional
from numbers import Integral

from .. import _ffi
from .base import PrimExpr, Span
from .expr import PrimExprWithOp, PrimVar, RangeExpr, PrimIterVar
from .stmt import Stmt
from .type import PointerType, PrimType

from ..runtime import Object
from . import _ffi_api, const
from ._converter import to_ir_object as _to_ir


@_ffi.register_object("ir.Buffer")
class Buffer(Object):
    """Symbolic data buffer in MATX.
    Buffer provide a way to represent data layout
    specialization of data structure in TVM.
    Do not construct directly, use :py:func:`~decl_buffer` instead.
    See the documentation of :py:func:`decl_buffer` for more details.
    See Also
    --------
    decl_buffer : Declare a buffer
    """

    READ = 1
    WRITE = 2

    def vload(self, begin, dtype=None):
        """Generate an Expr that loads dtype from begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        dtype : str
            The data type to be loaded,
            can be vector type which have lanes that is multiple of Buffer.dtype

        Returns
        -------
        load : Expr
            The corresponding load expression.
        """
        begin = (begin,) if isinstance(begin, (int, PrimExpr)) else begin
        dtype = dtype if dtype else self.dtype
        return _ffi_api.BufferVLoad(self, _to_ir(begin), dtype)  # type: ignore

    def vstore(self, begin, value):
        """Generate a Stmt that store value into begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        value : Expr
            The value to be stored.

        Returns
        -------
        store : Stmt
            The corresponding store stmt.
        """
        begin = (begin,) if isinstance(begin, (int, PrimExpr)) else begin
        return _ffi_api.BufferVStore(self, _to_ir(begin), value)  # type: ignore


def decl_buffer(
        shape,
        dtype=None,
        name="buffer",
        data=None,
        strides=None,
        elem_offset=None,
        scope="",
        data_alignment=-1,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
        span=None,
):
    """Declare a new symbolic buffer.
    Normally buffer is created automatically during lower and build.
    This is only needed if user want to specify their own buffer layout.
    See the note below for detailed discussion on usage of buffer.
    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.
    dtype : str, optional
        The data type of the buffer.
    name : str, optional
        The name of the buffer.
    data : Var, optional
        The data pointer in the buffer.
    strides: array of Expr
        The stride of the buffer.
    elem_offset: Expr, optional
        The beginning offset of the array to data.
        In terms of number of elements of dtype.
    scope: str, optional
        The storage scope of the buffer, if not global.
        If scope equals empty string, it means it is global memory.
    data_alignment: int, optional
        The alignment of data pointer in bytes.
        If -1 is passed, the alignment will be set to TVM's internal default.
    offset_factor: int, optional
        The factor of elem_offset field, when set,
        elem_offset is required to be multiple of offset_factor.
        If 0 is pssed, the alignment will be set to 1.
        if non-zero is passed, we will created a Var for elem_offset if elem_offset is not None.
    buffer_type: str, optional, {"", "auto_broadcast"}
        auto_broadcast buffer allows one to implement broadcast computation
        without considering whether dimension size equals to one.
        TVM maps buffer[i][j][k] -> buffer[i][0][k] if dimension j's shape equals 1.
    axis_separators : list of int, optional
        If passed, a list of separators between groups of axes,
        each of which is flattened to an output axis.  For flat
        memory spaces, should either be None, or an empty list.
    span: Optional[Span]
        The location of the decl_buffer creation in the source.
    Returns
    -------
    buffer : matx.ir.Buffer
        The created buffer
    Example
    -------
    Here's an example of how broadcast buffer can be used to define a symbolic broadcast operation,
    .. code-block:: python
        # TODO
        pass
    Note
    ----
    Buffer data structure reflects the DLTensor structure in dlpack.
    While DLTensor data structure is very general, it is usually helpful
    to create function that only handles specific case of data structure
    and make compiled function benefit from it.
    If user pass strides and elem_offset is passed as None
    when constructing the function, then the function will be specialized
    for the DLTensor that is compact and aligned.
    If user pass a fully generic symbolic array to the strides,
    then the resulting function becomes fully generic.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    dtype = "float32" if dtype is None else dtype
    strides = () if strides is None else strides

    if axis_separators is None:
        axis_separators = []

    if offset_factor != 0 and elem_offset is None:
        shape_dtype = shape[0].dtype if shape and hasattr(shape[0], "dtype") else "int32"
        elem_offset = PrimVar("%s_elem_offset" % name, shape_dtype)
    if data is None:
        # Bool is represented as uint1 in the IR, but stored as int8
        storage_type = PrimType(dtype)
        storage_type = PrimType("int8") if storage_type.dtype == "bool" else storage_type
        data = PrimVar(name, PointerType(storage_type), span)
    return _ffi_api.Buffer(  # type: ignore
        data,
        dtype,
        _to_ir(shape),
        _to_ir(strides),
        _to_ir(elem_offset),
        _to_ir(name),
        data_alignment,
        offset_factor,
        buffer_type,
        _to_ir(axis_separators),
        span,
    )


@_ffi.register_object("ir.BufferRegion")
class BufferRegion(Object):
    """BufferRegion node.
    Parameters
    ----------
    buffer : Buffer
        The buffer of the buffer region
    region : List[RangeExpr]
        The region array of the buffer region
    """

    def __init__(self, buffer: Buffer, region: List[RangeExpr]):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferRegion, buffer, _to_ir(region))  # type: ignore


@_ffi.register_object("ir.MatchBufferRegion")
class MatchBufferRegion(Object):
    """MatchBufferRegion node.
    Parameters
    ----------
    buffer : Buffer
        The target buffer
    source : BufferRegion
        The region of source buffer
    """

    buffer: Buffer
    source: BufferRegion

    def __init__(self, buffer: Buffer, source: BufferRegion):
        self.__init_handle_by_constructor__(
            _ffi_api.MatchBufferRegion, buffer, source  # type: ignore
        )


@_ffi.register_object("ir.BufferLoad")
class BufferLoad(PrimExprWithOp):
    """Buffer load node.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be loaded.

    indices : List[PrimExpr]
        The buffer indices.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, indices, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferLoad, buffer, indices, span  # type: ignore
        )


@_ffi.register_object("ir.BufferStore")
class BufferStore(Stmt):
    """Buffer store node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    value : PrimExpr
        The value we to be stored.

    indices : List[PrimExpr]
        The indices location to be stored.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, value, indices, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferStore, buffer, value, indices, span  # type: ignore
        )


@_ffi.register_object("ir.ComputeBlock")
class ComputeBlock(Stmt):
    """ComputeBlock node.
    Parameters
    ----------
    iter_vars : List[IterVar]
        The block Variable.
    reads : List[BufferRegion]
        The read buffer regions of the block.
    writes: List[BufferRegion]
        The write buffer regions of the block.
    name_hint: str
        the name_hint of the block.
    body: Stmt
        The body of the block.
    init: Optional[Stmt]
        The init block of the reduction block
    alloc_buffers: Optional[list[Buffer]]
        The buffer allocations
    match_buffers: Optional[List[MatchBufferRegion]]
        The subregion buffer match
    annotations: Optional[Mapping[str, Object]]
        Additional annotation hints.
    span : Optional[Span]
        The location of this block in the source code.
    """

    iter_vars: List[PrimIterVar]
    reads: List[BufferRegion]
    writes: List[BufferRegion]
    name_hint: str
    body: Stmt
    init: Optional[Stmt]
    alloc_buffers: Optional[List[Buffer]]
    match_buffers: Optional[List[MatchBufferRegion]]
    annotations: Optional[Mapping[str, Object]]
    span: Optional[Span]

    def __init__(
            self,
            iter_vars: List[PrimIterVar],
            reads: List[BufferRegion],
            writes: List[BufferRegion],
            name_hint: str,
            body: Stmt,
            init: Optional[Stmt] = None,
            alloc_buffers: Optional[List[Buffer]] = None,
            match_buffers: Optional[List[MatchBufferRegion]] = None,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
    ):
        if alloc_buffers is None:
            alloc_buffers = []
        if match_buffers is None:
            match_buffers = []
        if annotations is None:
            annotations = {}
        self.__init_handle_by_constructor__(
            _ffi_api.ComputeBlock,  # type: ignore
            _to_ir(iter_vars),
            _to_ir(reads),
            _to_ir(writes),
            name_hint,
            body,
            init,
            _to_ir(alloc_buffers),
            _to_ir(match_buffers),
            _to_ir(annotations),
            span,
        )  # type: ignore


@_ffi.register_object("ir.ComputeBlockRealize")
class ComputeBlockRealize(Stmt):
    """ComputeBlockRealize node.
    Parameters
    ----------
    iter_values : List[PrimExpr]
        The binding values of the block var.
    predicate : Union[PrimExpr, bool]
        The predicate of the block.
    block : ComputeBlock
        The block to realize
    span : Optional[Span]
        The location of this block_realize in the source code.
    """

    def __init__(
            self,
            iter_values: List[PrimExpr],
            predicate: Union[PrimExpr, bool],
            block: ComputeBlock,
            span: Optional[Span] = None,
    ):
        if isinstance(predicate, bool):
            predicate = const(predicate, "bool")
        self.__init_handle_by_constructor__(
            _ffi_api.ComputeBlockRealize,  # type: ignore
            iter_values,
            predicate,
            block,
            span,
        )  # type: ignore
