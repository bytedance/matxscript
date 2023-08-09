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
from typing import Any, Dict, List, Union, TYPE_CHECKING

import numpy as np

import matx.kernel.graphIR as _gir
import matx.kernel.symbol.utils as symbol_utils
import matx.kernel.typing.utils as typing_utils
from matx.kernel.func_registery import FUNC_REGISTRY
from matx.kernel.parser.ast_visitor.general_ast_visitor import GeneralAstVisitor
from matx.kernel.parser.ast_visitor.loop_ast_visitor import LoopAstVisitor
from matx.kernel.typing import NDArrayType as kernelNDArrayT
from matx.script import context as script_context

from .function_parser import FunctionParser
from .utils import BodyIterator, FuncReturnKind

if TYPE_CHECKING:
    from matx.kernel.kernel_parser import KernelParser


class TemplateParser(FunctionParser):

    def __init__(self, kernel_p: 'KernelParser', node: script_context.ASTNode):
        super().__init__(kernel_p, node)
        # template does not need return ctx
        self.return_ctx = None
        self.func_return_kind = FuncReturnKind.TEMPLATE

    def check_return(self) -> Any:
        # can do nothing here since template has no annotation
        return
