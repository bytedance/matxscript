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

from matx.pipeline.jit_object import JitObject, JitOpImpl, FuncMeta


def get_kernel_func(dso_path: str, func_name: str, file_name: str, lineno: int) -> JitOpImpl:
    dso_path_cxx11 = ""
    meta_info = FuncMeta(func_name, False, [], [])
    function_mapping = {func_name: func_name}
    share = False
    captures = []

    jit_obj = JitObject(
        dso_path=dso_path,
        dso_path_cxx11=dso_path_cxx11,
        meta_info=meta_info,
        function_mapping=function_mapping,
        share=share,
        captures=captures,
        py_source_file=file_name.encode(),
        py_source_line=lineno,
    )
    return JitOpImpl(func_name, jit_obj)
