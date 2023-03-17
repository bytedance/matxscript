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

from .symbol import is_symbol
from .typing import NDArrayType as kernelNDArrayT
from .typing import is_ndarray
from .. import ir as _ir
from ..ir.tensor_stmt import decl_buffer


class NDArrayContext:
    def __init__(self, name: str, type_: kernelNDArrayT, shape_symbol_table: dict, span) -> None:
        assert is_ndarray(type_), 'syntax error'
        self.name: str = name
        self.kernel_type: kernelNDArrayT = type_  # NDARRAY TYPE
        self.shape = type_.shape
        self.script_type = _ir.PointerType(_ir.PrimType(type_.dtype_str()))
        self.ndarray_var = _ir.HLOVar(name, self.script_type, span)  # HLO_VAR
        self.data_var = _ir.PrimVar(name, type_.dtype_str(), span)  # PRIM_VAR
        buffer_shape = [dim if not is_symbol(dim) else shape_symbol_table[dim]
                        for dim in self.shape]
        self.buffer = decl_buffer(buffer_shape)


class SymbolContext:

    def __init__(self, name: str):
        pass
