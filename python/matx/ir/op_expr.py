# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the Op is inspired by incubator-tvm.
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

"""Base Operator Expr used in IR expression."""
from .. import _ffi
from . import _ffi_api
from .base import HLOExpr
from ._converter import to_ir_object as _to_ir


@_ffi.register_object("Op")
class Op(HLOExpr):
    """Primitive operator in the IR."""

    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    @staticmethod
    def get(op_name):
        """Get the Op for a given name

        Parameters
        ----------
        op_name : str
            The operator name

        Returns
        -------
        op : Op
            The op of the corresponding name
        """
        return _ffi_api.GetOp(_to_ir(op_name))

    def get_attr(self, attr_name):
        """Get additional attribute about the operator.

        Parameters
        ----------
        attr_name : str
            The attribute name.

        Returns
        -------
        value : object
            The attribute value
        """
        return _ffi_api.OpGetAttr(self, _to_ir(attr_name))

    def set_attr(self, attr_name, value, plevel=10):
        """Set attribute about the operator.

        Parameters
        ----------
        attr_name : str
            The attribute name

        value : object
            The attribute value

        plevel : int
            The priority level
        """
        _ffi_api.OpSetAttr(self, _to_ir(attr_name), _to_ir(value), plevel)

    def reset_attr(self, attr_name):
        """Reset attribute about the operator.

        Parameters
        ----------
        attr_name : str
            The attribute name
        """
        _ffi_api.OpResetAttr(self, _to_ir(attr_name))
