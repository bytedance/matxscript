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

import ast
import inspect

from matx.script import context as script_context
import matx.kernel.typing.utils as typing_utils


def parse_ast(func):
    src_code = inspect.getsource(func)
    src_file_path = inspect.getfile(func)
    source_file_content, src_line_number = inspect.findsource(func)
    src_ast = ast.parse(src_code)
    ast.increment_lineno(src_ast, src_line_number)

    return src_ast, src_file_path, src_line_number, src_code


def extract_symbol_from_type(t):
    shape = t.shape
    symbols = set([dim for dim in shape if typing_utils.is_symbol(dim)])
    return {str(s): s for s in symbols}


def user_function_wrapper(value, resource_handle, span):
    if isinstance(value, script_context.ASTNode) and inspect.isfunction(value.raw):
        raise SyntaxError("_make_user_function is not support yet")
    if isinstance(value, script_context.ASTNode) and inspect.isclass(value.raw):
        raise SyntaxError("_make_user_class_creator is not support yet")
    if isinstance(value, script_context.GetClassAttr):
        return value.as_user_function(resource_handle, span)
    return value
