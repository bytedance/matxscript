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
from matx.kernel.typing import *
from matx.kernel.typing import NDArrayType as kernelNDArrayT
from .base import *


class ScalarNode(ExpressionBaseNode):
    def __init__(self, name: str, type_: kernelNDArrayT, span):
        super().__init__(type_)
        assert is_scalar_type(type_), 'syntax error'
        self.shape = type_.shape
        assert is_scalar_shape(self.shape), 'sytax error'
        self.script_type = _ir.PrimType(type_.dtype_str())
        self.name: str = name
        self.script_var = _ir.PrimVar(name, self.script_type, span)

    def to_matx_ir(self, **kwargs):
        return self.script_var


class ConstScalarNode(ScalarNode):

    def __init__(self, value, type_: kernelNDArrayT, span):
        super().__init__("const", type_, span)
        self.script_var = _ir.const(value, type_.dtype_str())


class IterScalarNode(ScalarNode):
    def __init__(self, name: str, type_: kernelNDArrayT, start, end, step, span):
        super().__init__(name, type_, span)
        self.start = start
        self.end = end
        self.step = step

    def to_matx_ir(self, key=None, **kwargs):
        if key == "start":
            return self.start.to_matx_ir()
        if key == "end":
            return self.end.to_matx_ir()
        if key == "step":
            return self.step.to_matx_ir()
        return super().to_matx_ir(**kwargs)
