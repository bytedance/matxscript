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
from matx._typed_ast import ast
from .. import context
from typing import Optional

from .script_error import MATXScriptError
from ..context import Span


def raise_syntax_error(custom_node: context.ASTNode,
                       ast_node: ast.AST,
                       err_msg: str,
                       err_type: Exception = SyntaxError):
    root_span = custom_node.span
    abs_lineno = root_span.lineno + ast_node.lineno - 1
    source_code = root_span.source_code

    new_reporter_span = Span()
    new_reporter_span.file_name = root_span.file_name
    new_reporter_span.lineno = abs_lineno
    try:
        new_reporter_span.func_name = custom_node.context.name
    except:
        new_reporter_span.func_name = ''
    new_reporter_span.source_code = source_code
    raise MATXScriptError(err_msg, new_reporter_span, err_type)


class Reporter:
    @classmethod
    def error(cls,
              msg: str = 'InternalError',
              span: Optional[Span] = None,
              err_type: Exception = None
              ):
        raise MATXScriptError(msg, span, err_type)

    @classmethod
    def error_with_node(cls,
                        custom_node: context.ASTNode,
                        ast_node: ast.AST,
                        msg: str = 'InternalError',
                        err_type: Exception = None):
        root_span = custom_node.span
        abs_lineno = root_span.lineno + ast_node.lineno - 1
        source_code = root_span.source_code

        new_reporter_span = Span()
        new_reporter_span.file_name = root_span.file_name
        new_reporter_span.lineno = abs_lineno
        new_reporter_span.func_name = custom_node.context.name
        new_reporter_span.source_code = source_code
        raise MATXScriptError(msg, new_reporter_span, err_type)
