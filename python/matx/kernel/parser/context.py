#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import sympy

from matx import ir as _ir
from matx.ir.tensor_stmt import decl_buffer
from matx.kernel.typing import *
from matx.kernel.typing import NDArrayType as kernelNDArrayT
from ..symbol import is_symbol_type


class AbstractBaseVariableContext:

    def __init__(self, type_) -> None:
        self.kernel_type = type_  # NDARRAY TYPE
        if is_scalar_type(type_):
            self.script_type = _ir.PrimType(type_.dtype_str())
        elif is_ndarray_type(type_):
            self.script_type = _ir.PointerType(_ir.PrimType(type_.dtype_str()))
        elif is_symbol_type(type_):
            self.script_type = _ir.PrimType("int64")
        else:
            raise SyntaxError(f"unknown type {type_}")
        self._abstract_ctx = True

    def is_abstract_ctx(self):
        return self._abstract_ctx


class AbstractNDArrayContext(AbstractBaseVariableContext):

    def __init__(self, type_: kernelNDArrayT) -> None:
        assert is_ndarray_type(type_), 'syntax error'
        super().__init__(type_)
        self.shape = type_.shape
        self.data = AbstractScalarContext(ScalarType(type_.dtype))


class AbstractScalarContext(AbstractBaseVariableContext):

    def __init__(self, type_: kernelNDArrayT):
        assert is_scalar_type(type_), 'syntax error'
        assert is_scalar_shape(type_.shape), 'syntax error'
        super().__init__(type_)
        self.shape = type_.shape
        self.data = None


class NDArrayContext(AbstractNDArrayContext):
    def __init__(self, name: str, type_: kernelNDArrayT, shape_symbol_table: dict, span) -> None:
        assert is_ndarray_type(type_), 'syntax error'
        super().__init__(type_)
        self.name: str = name
        self.script_var = _ir.PrimVar(f"{name}", self.script_type, span)  # PTR_VAR
        buffer_shape = [dim if not is_symbol(dim) else shape_symbol_table[str(dim)].script_var
                        for dim in self.shape]
        self.data = ScalarContext(f"{name}_data", ScalarType(type_.dtype), span)
        self.buffer = decl_buffer(
            buffer_shape,
            dtype=type_.dtype_str(),
            name=name,
            data=self.script_var)
        self._abstract_ctx = False


class ScalarContext(AbstractScalarContext):
    def __init__(self, name: str, type_: kernelNDArrayT, span):
        super().__init__(type_)
        assert is_scalar_type(type_), 'syntax error'
        assert is_scalar_shape(self.shape), 'sytax error'
        self.name: str = name
        self.script_type = _ir.PrimType(type_.dtype_str())
        self.script_var = _ir.PrimVar(name, self.script_type, span)
        self.data = None
        self._abstract_ctx = False


class ConstScalarContext(ScalarContext):

    def __init__(self, value, type_: kernelNDArrayT, span):
        super().__init__("const", type_, span)
        self.script_type = _ir.PrimType(type_.dtype_str())
        self.script_var = _ir.const(value, type_.dtype_str())


class SymbolContext(AbstractBaseVariableContext):

    def __init__(self, symbol, span) -> None:
        super().__init__(sympy.Basic)
        assert is_symbol(symbol), 'syntax error'
        self.name: str = str(symbol)
        self.script_var = _ir.PrimVar(f"symbol_{self.name}", "int64", span)
        self.data = None
