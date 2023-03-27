# Copyright 2022 ByteDance Ltd. and/or its affiliates.
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
"""Class IR."""

from .. import _ffi
from . import _ffi_api
from .base import Stmt, Span
from ._converter import to_ir_object as _to_ir


@_ffi.register_object("ir.ClassStmt")
class ClassStmt(Stmt):
    """A class stmt.

    Parameters
    ----------
    name: str
        Class Name.
    base: Optional[ClassStmt]
        The Base of this Class.
    body: List[ir.Stmt]
        The body of the Class.
    cls_type: ir.ClassType
        The type of the class.
    attrs: dict
        The attributes
    span: Span
        The source code info
    """

    def __init__(self,
                 name,
                 base,
                 body,
                 cls_type,
                 attrs=None,
                 span=Span()):
        self.__init_handle_by_constructor__(
            _ffi_api.ClassStmt,
            _to_ir(name),
            _to_ir(base),
            _to_ir(body),
            _to_ir(cls_type),
            _to_ir(attrs),
            _to_ir(span)
        )

    @property
    def attrs(self):
        """Return the attrs member of the function."""
        return _ffi_api.ClassStmt_Attrs(self)

    def with_attr(self, attr_key_or_dict, attr_value=None):
        """Create a new copy of the ClassStmt and update the attribute.

        Parameters
        ----------
        attr_key_or_dict : Union[str, dict]
            The attribute key to use or a dict containing multiple key value pairs.

        attr_value : Any
            The new attribute value.

        Returns
        -------
        func : ClassStmt
            A new copy of the ClassStmt
        """
        # make sure we first copy so that we can safely do copy on write
        # for multiple updates.
        res = _ffi_api.ClassStmt_Copy(self)

        if isinstance(attr_key_or_dict, dict):
            for key, val in attr_key_or_dict.items():
                res = _ffi_api.ClassStmt_WithAttr(res._move(), _to_ir(key), _to_ir(val))
            return res

        return _ffi_api.ClassStmt_WithAttr(
            res._move(), _to_ir(attr_key_or_dict), _to_ir(attr_value)
        )

    def get_type(self):
        return _ffi_api.ClassStmt_GetType(self)
