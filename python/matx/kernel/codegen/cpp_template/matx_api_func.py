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
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from matx.kernel.parser.utils import FuncReturnKind


class MatxAPIFuncCodegen:
    """Class for keeping track of data necessary for c++ interface func codegen.
    c++ interface:
        MATX_DLL {{rt_type}} {{python_func_name}}({{args}}, void* handle_2_71828182846)
    """

    def __init__(self, meta_data: MatxInterfaceCodegenMetaData) -> None:
        self.meta_data = meta_data

        self.type_cvt_code_list: List[str] = []
        self.call_func_code_list: list[str] = []
        self.ndarray_ptr_list: List[str] = []
        self.arg_type_and_name: str = ""
        self.mlir_args: List[str] = []

        self.gen_mlir_func_type()
        self.gen_ndarray_cvt_code()
        self.gen_mlir_func_call_code()
        self.arg_type_and_name = ", ".join([f"{t} {n}" for t, n in zip(
            self.meta_data.matx_arg_types, self.meta_data.matx_arg_names)])

    def func_declaration(self):
        return f"MATX_DLL auto {self.meta_data.matx_func_name}({', '.join(self.meta_data.matx_arg_types)} handle_2_71828182846 =((void*)(int64_t)0));"

    def gen_ndarray_cvt_code(self):
        i = 0
        for arg_type, arg_name in zip(self.meta_data.arg_types, self.meta_data.arg_names):
            if arg_type == "NDArray":
                c0 = f"auto && __func_arg{i}_shared_ptr = convert_from_ndarray({arg_name});"
                c1 = f"auto __func_arg{i} = __func_arg{i}_shared_ptr.get();"
                self.type_cvt_code_list.extend([c0, c1])
                self.mlir_args.append(f"__func_arg{i}")
                self.ndarray_ptr_list.append(f"__func_arg{i}")
            else:
                self.mlir_args.append(arg_name)
            i += 1

    def gen_mlir_func_type(self):
        if self.meta_data.func_return_kind.is_void():
            self.mlir_func_type = f"void(*)({', '.join(self.meta_data.mlir_arg_types)});"
        elif self.meta_data.func_return_kind.is_scalar():
            self.mlir_func_type = f"{self.meta_data.matx_rt_type}(*)({', '.join(self.meta_data.mlir_arg_types)});"
        elif self.meta_data.func_return_kind.is_dynamic_tensor():
            self.mlir_func_type = f"void(*)({', '.join(['void *', *self.meta_data.mlir_arg_types])});"
        else:
            raise SyntaxError(
                f"function_return_kind({self.meta_data.func_return_kind}) is not supported")

    def gen_mlir_func_call_code(self):
        if self.meta_data.func_return_kind.is_void():
            self.call_func_code_list = [
                f"casted_func_ptr({', '.join(self.mlir_args)});",
                "return None;"
            ]
        elif self.meta_data.func_return_kind.is_scalar():
            self.call_func_code_list = [
                f"return casted_func_ptr({', '.join(self.mlir_args)});"
            ]
        elif self.meta_data.func_return_kind.is_dynamic_tensor():
            rt_shared_ptr = "_mlir_return_31905_shared_ptr_571"
            rt_ptr = "_mlir_return_31905_ptr_571"
            self.call_func_code_list = [
                f"auto && {rt_shared_ptr} = alloc_memref_descriptor_ptr({self.meta_data.return_ndim});",
                f"void * {rt_ptr} = {rt_shared_ptr}.get();",
                f"casted_func_ptr({', '.join([rt_ptr, *self.mlir_args])});",
                f"""return convert_to_ndarray({rt_shared_ptr}, {self.meta_data.return_ndim}, cvt_str_to_dl_dtype("{self.meta_data.return_dtype}"));"""]
        else:
            raise SyntaxError(
                f"function_return_kind({self.meta_data.func_return_kind}) is not supported")

    def func_definition(self) -> str:
        func_definition_template = JINJA2_ENV.get_template('matx_api_func.txt')
        return func_definition_template.render(
            mlir_func_type=self.mlir_func_type,
            matx_func_name=self.meta_data.matx_func_name,
            arg_type_and_name=self.arg_type_and_name,
            mlir_func_name=self.meta_data.mlir_func_name,
            lib_path=self.meta_data.lib_path,
            type_cvt_code_list=self.type_cvt_code_list,
            call_func_code_list=self.call_func_code_list
        )
