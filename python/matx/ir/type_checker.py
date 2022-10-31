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

from .base import Span, BaseExpr
from . import type as _type
from . import type_relation as _type_rel


def handle_error(span: Span, err_msg: str, err_type: Exception = TypeError):
    err_context = 'File "{}", line {}, in {}'.format(
        span.file_name.decode(), span.lineno, span.func_name.decode())
    typed_err_msg = '{}: {}'.format(err_type.__name__, err_msg)
    err_info = err_context + "\n" + span.source_code.decode() + "\n" + typed_err_msg
    raise RuntimeError(err_info)


def check_int_or_generic(span: Span, input_expr: BaseExpr, msg_prefix: str = ""):
    err_msg = "{}'{}' object cannot be interpreted as an integer"
    int_or_generic_type = (_type.PrimType.BoolType, _type.PrimType.IntType, _type.ObjectType)
    if isinstance(msg_prefix, str) and len(msg_prefix) == 0:
        msg_prefix += ", "
    if not _type_rel.is_type_of(input_expr, int_or_generic_type):
        handle_error(
            span, err_msg.format(msg_prefix, input_expr.py_type_name())
        )
