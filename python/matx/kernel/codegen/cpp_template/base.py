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
import time
import jinja2
import numpy as np

import matx.kernel.typing.utils as typing_utils
from matx.kernel.typing import STR_TO_PYTYPE
from matx.kernel.parser.utils import FuncReturnKind

if TYPE_CHECKING:
    from matx.kernel.kernel_parser import KernelParser

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

    def __init__(
            self,
            file_name: str,
            line_no: int,
            lib_path: str,
            unique_id: int,
            py_func_name: str,
            func_return_kind: FuncReturnKind,
            arg_names: List[str],
            arg_types: List[str],
            rt_type: str,
            return_ndim: int,
            return_dtype: str) -> None:
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


PYTYPE_TO_CPP_TYPE_STR = {
    bool: "bool",
    int: "int32_t",
    float: "float",
    np.bool_: "bool",
    np.int8: "int8_t",
    np.int16: "int16_t",
    np.int32: "int32_t",
    np.int64: "int64_t",
    np.intc: "int32_t",
    np.uint8: "uint8_t",
    np.uint16: "uint16_t",
    np.uint32: "uint32_t",
    np.uint64: "uint64_t",
    np.uintc: "uint32_t",
    # todo support float16
    # np.float16 has no corresponding python builtin ctypes
    np.float16: "__fp16",
    np.float32: "float",
    np.float64: "double",
    np.longlong: "int64_t",
    np.ulonglong: "uint64_t"
}


def cvt_to_cpp_type_str(t):
    if typing_utils.is_scalar_type(t):
        return PYTYPE_TO_CPP_TYPE_STR[t.dtype]
    elif typing_utils.is_ndarray_type(t):
        return "NDArray"
    elif typing_utils.is_symbol(t):
        return "int64_t"
    else:
        raise SyntaxError(f"Unsupported type {t}")


def make_meta_data(parser: 'KernelParser', lib_path: str) -> MatxInterfaceCodegenMetaData:
    file_name: str = parser.file_name
    line_no: int = parser.line_no

    _nanoseconds = int(time.time() * 1e9)
    unique_id: int = int(_nanoseconds / 100) + 0x01b21dd213814000
    python_func_name: str = parser.func_name
    func_return_kind: FuncReturnKind = parser.graph.func_return_kind

    arg_names: List[str] = [k for k in parser.args.keys()]
    arg_types: List[str] = [cvt_to_cpp_type_str(t) for t in parser.arg_types]

    if func_return_kind.is_void():
        return_type: str = "void"
    elif func_return_kind.is_scalar():
        return_type: str = PYTYPE_TO_CPP_TYPE_STR[STR_TO_PYTYPE[parser.graph.return_dtype_str]]
    elif func_return_kind.is_dynamic_tensor():
        return_type: str = "void"
    else:
        raise SyntaxError(f"Unsupported return type {func_return_kind}")

    return_ndim: int = len(parser.graph.return_shape)
    return_dtype: str = parser.graph.return_dtype_str
    return MatxInterfaceCodegenMetaData(
        file_name, line_no, lib_path, unique_id, python_func_name, func_return_kind,
        arg_names, arg_types, return_type, return_ndim, return_dtype
    )
