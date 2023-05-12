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

from matx.ir import _ffi_node_api
import subprocess, os


def compile_linalg(matx_ir, file_name = "tmp.mlir"):
    code = _ffi_node_api.as_linalg_text(matx_ir).decode()
    with open(file_name, "w+") as f:
        f.write(code)
    env = os.environ.copy()
    process = subprocess.Popen(['mlir-opt',
                                '--convert-linalg-to-loops',
                                '--lower-affine',
                                '--convert-scf-to-cf',
                                '--convert-linalg-to-llvm',
                                '--convert-func-to-llvm',
                                '--convert-index-to-llvm',
                                '--convert-arith-to-llvm',
                                '--convert-memref-to-llvm',
                                '--convert-cf-to-llvm',
                                '--reconcile-unrealized-casts',
                                file_name,
                                '-o',
                                'llvm_'+file_name],
                               env = env,
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    err = stderr.decode()
    if len(err) != 0:
        raise RuntimeError("\n"+err)
