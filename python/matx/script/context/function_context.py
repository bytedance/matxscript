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
from typing import Dict, List, Optional
from enum import Enum, auto

from .typing import PrimValueType, AnnotatedType
from ... import ir as _ir


class FunctionType(Enum):
    FUNCTION = auto()
    INSTANCE = auto()
    CLASS = auto()
    STATIC = auto()


class FunctionContext:

    def __init__(
            self,
            fn_name: str = '<unknown>',
            return_type: Optional[AnnotatedType] = None,
            arg_names: Optional[List[str]] = None,
            arg_types: Optional[Dict[str, AnnotatedType]] = None,
            arg_defaults: Optional[Dict[str, PrimValueType]] = None,
            arg_reassigns: Optional[Dict[str, bool]] = None,
            fn_type: Optional[FunctionType] = None,
            is_abstract: bool = False
    ):
        self.raw: Optional[type] = None
        self.fn_name = fn_name
        self.return_type = return_type  # Deferred?
        self.arg_names = arg_names or []
        self.arg_types = arg_types or {}  # Deferred?
        self.arg_defaults = arg_defaults or {}  # Deferred?
        self.arg_reassigns = arg_reassigns or {}
        self.fn_type = fn_type
        self.is_abstract = is_abstract

        self.unbound_name: str = fn_name

    @property
    def name(self):
        return self.fn_name

    def __str__(self):
        args = [
            (f'{arg_name}: {self.arg_types[arg_name]}' if arg_name in self.arg_types else arg_name)
            for arg_name in self.arg_names
        ]
        if self.fn_type is FunctionType.INSTANCE:
            args = ['self'] + args
        return ''.join([
            f'def {self.fn_name}(',
            ', '.join(args),
            f') -> {self.return_type}: ...',
        ])

    def to_ir_schema(self) -> _ir.FuncType:
        arg_types = [ty for _, ty in self.arg_types.items()]
        func_type = _ir.FuncType(arg_types, self.return_type)
        return func_type
