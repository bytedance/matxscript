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
import jinja2
import os
import time
import numpy as np

import matx.kernel.typing.utils as typing_utils
from matx.kernel.typing import STR_TO_PYTYPE

TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))

if TYPE_CHECKING:
    from matx.kernel.parser.utils import FuncReturnKind
    from matx.kernel.kernel_parser import KernelParser

DEBUG = False


@dataclass
class CInterfaceCodegenData:
    """Class for keeping track of data necessary for c++ interface codegen."""
    unique_id: int
    func_name: str
    return_type: str
    return_ndim: int
    return_dtype: str
    input_types: List[str]
    lib_path: str
    func_return_kind: 'FuncReturnKind'
    free_return: bool
    debug: bool

    def __init__(self, unique_id: int, func_name: str, return_type: str, return_ndim: int,
                 return_dtype: str, input_types: List[str], lib_path: str,
                 func_return_kind: 'FuncReturnKind', debug: bool = False):
        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))
        self.template = self.env.get_template('cpp_header.txt')
        self.unique_id = unique_id
        self.func_name = func_name
        self.return_type = return_type
        self.return_ndim = return_ndim
        self.return_dtype = return_dtype
        self.input_types = input_types
        self.lib_path = lib_path
        self.func_return_kind = func_return_kind
        self.debug = DEBUG or debug

    def code(self):
        output = self.template.render(unique_id=self.unique_id,
                                      func_name=self.func_name,
                                      return_type=self.return_type,
                                      return_ndim=self.return_ndim,
                                      return_dtype=self.return_dtype,
                                      input_types=self.input_types,
                                      lib_path=self.lib_path,
                                      func_return_kind=self.func_return_kind,
                                      debug=self.debug)
        return output


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
        return "void *"
    elif typing_utils.is_symbol(t):
        return "int64_t"
    else:
        raise SyntaxError(f"Unsupported type {t}")


def from_kernel_parser(parser: 'KernelParser', lib_path: str) -> CInterfaceCodegenData:
    nanoseconds = int(time.time() * 1e9)
    unique_id: int = int(nanoseconds / 100) + 0x01b21dd213814000
    func_name: str = parser.func_name
    return_ndim: int = len(parser.graph.return_shape)
    return_dtype: str = parser.graph.return_dtype_str
    input_types: List[str] = [cvt_to_cpp_type_str(t) for t in parser.arg_types]
    func_return_kind: 'FuncReturnKind' = parser.graph.func_return_kind
    if func_return_kind.is_void():
        return_type: str = "void"
    elif func_return_kind.is_scalar():
        return_type: str = PYTYPE_TO_CPP_TYPE_STR[STR_TO_PYTYPE[parser.graph.return_dtype_str]]
    elif func_return_kind.is_dynamic_tensor():
        return_type: str = "void"
    else:
        raise SyntaxError(f"Unsupported return type {func_return_kind}")
    return CInterfaceCodegenData(unique_id, func_name, return_type, return_ndim,
                                 return_dtype, input_types, lib_path, func_return_kind)
