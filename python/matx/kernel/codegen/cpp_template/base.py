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
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import os
import jinja2


if TYPE_CHECKING:
    from matx.kernel.parser.utils import FuncReturnKind


TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
JINJA2_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))


@dataclass
class MatxInterfaceCodegenMetaData:
    """Class for keeping track of data necessary for matx interface func codegen."""
    file_name: str
    line_no: int
    lib_path: str

    unique_id: int
    python_func_name: str
    func_return_kind: FuncReturnKind

    arg_names: List[str]
    arg_types: List[str]
    arg_len: int

    mlir_func_name: str
    mlir_arg_types: List[str]

    matx_func_name: str
    matx_arg_types: List[str]
    matx_arg_names: List[str]
    matx_rt_type: str

    c_api_func_name: str
    c_api_arg_types: List[str]

    return_ndim: int
    return_dtype: str

    def __init__(self, file_name: str, line_no: int, lib_path: str,
                 unique_id: int, py_func_name: str, func_return_kind: FuncReturnKind,
                 arg_names: List[str], arg_types: List[str], rt_type: str, return_ndim: int, return_dtype: str) -> None:
        self.file_name = file_name
        self.line_no = line_no
        self.lib_path = lib_path

        self.unique_id = unique_id
        self.python_func_name = py_func_name
        self.func_return_kind = func_return_kind

        self.arg_names = arg_names
        self.arg_types = arg_types
        self.arg_len = len(arg_names)

        self.mlir_func_name = f"_mlir_ciface_{py_func_name}"
        self.mlir_arg_types = [t if t != "NDArray" else "void *" for t in arg_types]

        self.matx_func_name = f"_{unique_id}_{py_func_name}_"
        self.matx_arg_types = [*arg_types, 'void *']
        self.matx_arg_names = [*arg_names, "handle_2_71828182846"]
        self.matx_rt_type = rt_type

        self.c_api_func_name = f"{self.matx_func_name}__c_api"

        self.return_ndim = return_ndim
        self.return_dtype = return_dtype



