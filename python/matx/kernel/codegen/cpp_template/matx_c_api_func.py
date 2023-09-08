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

from .base import MatxInterfaceCodegenMetaData, JINJA2_ENV


class MatxCAPIFuncCodegen:
    """Class for keeping track of data necessary for c interface func codegen.
    c interface:
        int {{python_func_name}}__c_api(MATXScriptAny* , int , MATXScriptAny* , void*)
    """

    def __init__(self, meta_data: MatxInterfaceCodegenMetaData) -> None:
        self.meta_data = meta_data
        self.pos_args_cvt_code = []
        self.args_t_cvt_code = []
        self.gen_cvt_code()

    def func_declaration(self) -> str:
        return f"int {self.meta_data.c_api_func_name}(MATXScriptAny* , int , MATXScriptAny* , void*);"

    def gen_cvt_code(self):
        arg_cvt_template = JINJA2_ENV.get_template('matx_c_api_func_arg_cvt.txt')
        for arg_idx, arg_type in enumerate(self.meta_data.arg_types):
            pos_rd = arg_cvt_template.render(
                arg_name="pos_args",
                arg_type=arg_type,
                arg_idx=arg_idx,
                file_name=self.meta_data.file_name,
                line_no=self.meta_data.line_no,
                python_func_name=self.meta_data.python_func_name
            )
            args_rd = arg_cvt_template.render(
                arg_name="args_t",
                arg_type=arg_type,
                arg_idx=arg_idx,
                file_name=self.meta_data.file_name,
                line_no=self.meta_data.line_no,
                python_func_name=self.meta_data.python_func_name
            )

            self.pos_args_cvt_code.append(pos_rd)
            self.args_t_cvt_code.append(args_rd)
        self.pos_args_cvt_code.append("resource_handle")
        self.args_t_cvt_code.append("resource_handle")

    def func_definition(self) -> str:
        func_definition_template = JINJA2_ENV.get_template('matx_c_api_func.txt')
        return func_definition_template.render(
            c_api_func_name=self.meta_data.c_api_func_name,
            arg_len=self.meta_data.arg_len,
            arg_name_str=", ".join([f'"{a}"' for a in self.meta_data.arg_names]),
            matx_func_name=self.meta_data.matx_func_name,
            py_func_name=self.meta_data.python_func_name,
            pos_args_cvt_code=", ".join(self.pos_args_cvt_code),
            args_t_cvt_code=", ".join(self.args_t_cvt_code),
            file_name=self.meta_data.file_name,
            line_no=self.meta_data.line_no,
        )
