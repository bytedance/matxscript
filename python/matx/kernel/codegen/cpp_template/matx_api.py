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


from .base import make_meta_data, MatxInterfaceCodegenMetaData, JINJA2_ENV
from .matx_api_func import MatxAPIFuncCodegen
from .matx_c_api_func import MatxCAPIFuncCodegen
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from matx.kernel.kernel_parser import KernelParser


class MatxApiCodegen:
    """Class for keeping track of data necessary for c++ interface codegen."""

    def __init__(self, meta_data: MatxInterfaceCodegenMetaData) -> None:
        self.meta_data = meta_data
        self.matx_api_func_code_gen = MatxAPIFuncCodegen(meta_data)
        self.matx_c_api_func_code_gen = MatxCAPIFuncCodegen(meta_data)

    def func_name(self):
        return self.meta_data.python_func_name

    def code(self):
        code_template = JINJA2_ENV.get_template('matx_api.txt')
        matx_api_declaration = self.matx_api_func_code_gen.func_declaration()
        matx_c_api_declaration = self.matx_c_api_func_code_gen.func_declaration()
        matx_api_definition = self.matx_api_func_code_gen.func_definition()
        matx_c_api_definition = self.matx_c_api_func_code_gen.func_definition()
        return code_template.render(
            matx_api_declaration=matx_api_declaration,
            matx_c_api_declaration=matx_c_api_declaration,
            matx_api_definition=matx_api_definition,
            matx_c_api_definition=matx_c_api_definition,
            c_interface_func_name=self.meta_data.c_api_func_name,
            py_func_name=self.meta_data.python_func_name
        )


def render_matx_api_code(parser: 'KernelParser',
                         lib_path: str) -> Tuple[str,
                                                 MatxInterfaceCodegenMetaData]:
    meta_data = make_meta_data(parser, lib_path)
    codegen_class = MatxApiCodegen(meta_data)
    return codegen_class.code(), meta_data
