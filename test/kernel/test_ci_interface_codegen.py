#  // Copyright 2023 ByteDance Ltd. and/or its affiliates.
#  /*
#   * Licensed to the Apache Software Foundation (ASF) under one
#   * or more contributor license agreements.  See the NOTICE file
#   * distributed with this work for additional information
#   * regarding copyright ownership.  The ASF licenses this file
#   * to you under the Apache License, Version 2.0 (the
#   * "License"); you may not use this file except in compliance
#   * with the License.  You may obtain a copy of the License at
#   *
#   *   http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing,
#   * software distributed under the License is distributed on an
#   * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   * KIND, either express or implied.  See the License for the
#   * specific language governing permissions and limitations
#   * under the License.
#   */


import unittest

from matx.kernel.parser import FuncReturnKind
from matx.kernel.codegen.cpp_template.function_meta_data import CInterfaceCodegenData


class TestBroadCast(unittest.TestCase):

    def test_basic_kernel_c_interface_codegen_void(self):

        unique_id = 123432423
        func_name = "func_name"
        return_type = "void *"
        return_ndim = 2
        return_dtype = "int8"
        input_types = ["void *", "int"]
        lib_path = "./"
        func_return_kind = FuncReturnKind.VOID

        func_meta = CInterfaceCodegenData(unique_id, func_name, return_type, return_ndim,
                                          return_dtype, input_types, lib_path, func_return_kind)
        print(func_meta.code())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
