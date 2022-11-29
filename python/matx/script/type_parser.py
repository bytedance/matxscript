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

from . import context
from .reporter import MATXScriptError
from ..ir.type_converter import _AnnTypeConvert, TypeConvertException


def parse_type(type_node: ast.AST, node: context.ASTNode, sc_ctx: context.ScriptContext):
    converter = _AnnTypeConvert(node.module)

    for dep_node in node.deps:
        ty_name = dep_node.context.name
        if ty_name in converter.ty_map:
            raise TypeError(f'internal error: {ty_name} is already in ty_map!')
        # converter.ty_map[ty_name] = dep_node.ir_schema
        ty_raw = dep_node.context.raw
        if ty_raw in converter.ty_map:
            raise TypeError(f'internal error: {ty_raw} is already in ty_map!')
        converter.ty_map[ty_raw] = dep_node.ir_schema

    """ Parse type """
    if type_node is None:
        sc_ctx.reporter.error_with_node(node, node.ast, "missing type annotation", SyntaxError)
    if not isinstance(type_node, ast.AST):
        sc_ctx.reporter.error_with_node(node, node.ast, "internal error", RuntimeError)
    try:
        ty = converter.convert(type_node)
        return ty
    except MATXScriptError:
        raise
    except TypeConvertException as e:
        sc_ctx.reporter.error_with_node(node, e.node, str(e), type(e))
    except BaseException as e:
        sc_ctx.reporter.error_with_node(node, type_node, str(e), TypeError)
