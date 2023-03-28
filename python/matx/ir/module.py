# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the IRModule is inspired by incubator-tvm.
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
"""IRModule that holds the functions and type definitions."""
from typing import Optional, List

from .. import _ffi
from . import _ffi_api

from .base import Node, Stmt
from ._converter import to_ir_object as _to_ir


@_ffi.register_object("ir.IRModule")
class IRModule(Node):
    """IRModule that holds functions and classes.

    IRModule is the basic unit for all IR transformations across the stack.

    Parameters
    ----------
    body: list of Stmt, optional
    """

    def __init__(self, body: Optional[List[Stmt]] = None):
        if body is None:
            body = []
        self.__init_handle_by_constructor__(
            _ffi_api.IRModule,
            _to_ir(body),
        )

    def __getitem__(self, name):
        """Lookup a global definition by name.
        Parameters
        ----------
        name: str
            The name or global variable.
        Returns
        -------
        val: Union[Function, Class]
            The definition referenced by :code:`var` (either a function or class).
        """
        assert isinstance(name, (bytes, bytearray, str))
        return _ffi_api.Module_Lookup(self, _to_ir(name))

    def add(self, val):
        assert isinstance(val, Stmt)
        _ffi_api.Module_Add(self, val)

    def update(self, other):
        """Insert functions in another Module to current one.

        Parameters
        ----------
        other: IRModule
            The module to merge into the current Module.
        """
        if isinstance(other, (list, tuple)):
            other = IRModule(other)

        return _ffi_api.Module_Update(self, other)

    def add_export_func(self, export_func):
        # TODO: optimize
        return _ffi_api.Module_AddExportFunction(self, export_func)

    def set_main(self, main):
        self.add_export_func(main)
