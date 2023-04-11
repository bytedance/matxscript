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
from matx import ir as _ir
from .base import *
from ..symbol import *


class SymbolNode(ExpressionBaseNode):

    def __init__(self, symbol, span) -> None:
        super().__init__(sympy.Basic)
        assert is_symbol(symbol), 'syntax error'
        self.name: str = str(symbol)
        self.script_type = _ir.PrimType("int64")
        self.script_var = _ir.PrimVar(f"symbol_{self.name}", "int64", span)

    def to_matx_ir(self, **kwargs):
        return self.script_var
